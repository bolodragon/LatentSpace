"""CTF-aware CWF for tam_dimer_tetramer, using noise_restacked.mrc.

Run from the project root with the project environment activated:
    python LatentSpace/experiment.py
    python LatentSpace/experiment.py --count 50 --resolution 64
    python LatentSpace/experiment.py --seed 1  # Choose different random subsets.

Dependencies: python -m pip install "aspire==0.14.3" mrcfile starfile matplotlib

The defaults randomly select 4,000 particles and 500 empty noise images without
replacement, using seed 0 for reproducibility. Output is 128 pixels, with batches
of 4, two threads, and 32 representative CTF groups.
Only requested image batches are copied out of the memory-mapped MRC stack.
Outputs include a denoised MRC stack, metadata CSV, preview PNGs, and run.json.
Each run is saved in PROJECT_ROOT/outputs/cwf_*/, alongside LatentSpace.

This retains the notebook's approximations: grouped/angularly averaged CTFs,
radial noise statistics, and matched reflection padding followed by cropping.
Ringing can remain; a successful run does not establish denoising accuracy.
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Save figures without creating a GUI or notebook outputs.
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import starfile
from aspire.basis import FFBBasis2D
from aspire.image import Image
from aspire.source import ImageSource
from aspire.operators import CTFFilter, ArrayFilter, PowerFilter, MultiplicativeFilter
from aspire.denoising import DenoiserCov2D
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "LatentSpace_data"


class SelectedMRCStack:
    """Expose selected stack indices while keeping only requested batches in RAM."""

    def __init__(self, data, indices):
        if data.ndim != 3 or data.shape[1] != data.shape[2]:
            raise ValueError("Expected a stack of square particle images")
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        if np.any(self.indices < 0) or np.any(self.indices >= len(data)):
            raise IndexError("STAR references an image outside the particle stack")
        self.shape = (len(self.indices), *data.shape[1:])
        self.ndim = 3
        self.dtype = data.dtype
        # Logical size: conservatively include potential mapped pages in preflight.
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        return self.data[self.indices[selection]]


def select_particles(star_path, stack_path, count, selection="random", seed=0,
                     stack_reference=None):
    """Keep image references and metadata in exactly the same order."""
    tables = starfile.read(star_path, always_dict=True)
    particles = tables["particles"]
    references = particles["rlnImageName"].astype(str).str.split("@", n=1, expand=True)
    if references.shape[1] != 2 or references[1].isna().any():
        raise ValueError("Invalid rlnImageName: expected index@stack_path")
    if stack_reference is None:
        matches = references[1].map(lambda value: Path(value).name == stack_path.name)
        paths = references.loc[matches, 1].unique()
        if len(paths) > 1:
            raise ValueError("Multiple exported stacks share this basename; specify --stack-reference")
    else:
        matches = references[1] == stack_reference
    selected = particles.loc[matches].copy()
    if selected.empty:
        raise ValueError("No STAR rows match the supplied particle stack")
    if count > len(selected):
        print(f"Requested {count} particles; using all {len(selected)} matching rows.", flush=True)
    count = min(count, len(selected))
    if count < 2:
        raise ValueError("At least two matching particles are required")
    if selection == "random":
        # Sort the sampled positions to preserve the original STAR ordering.
        positions = np.sort(np.random.default_rng(seed).choice(len(selected), count, replace=False))
        selected = selected.iloc[positions]
    else:
        selected = selected.iloc[:count]
    selected = selected.reset_index(drop=True)
    indices = selected["rlnImageName"].str.split("@", n=1).str[0].astype(np.int64).to_numpy() - 1
    return selected, tables["optics"], indices


def _cwf_downsample_batch(images, resolution):
    """Convert only the current batch to float32 and Fourier-downsample it."""
    batch = np.asarray(images, dtype=np.float32)
    if not np.isfinite(batch).all():
        raise ValueError("Images contain non-finite values")
    if batch.shape[-1] == resolution:
        return batch.copy()
    return Image(batch).downsample(resolution).asnumpy().copy()


def _cwf_padding(resolution, padding_factor):
    """Return equal margins so the complete output square lies inside the basis disk."""
    if not np.isfinite(padding_factor) or padding_factor <= np.sqrt(2):
        raise ValueError("padding_factor must exceed sqrt(2) to retain all square corners")
    margin = int(np.ceil((padding_factor - 1) * resolution / 2))
    return margin, resolution + 2 * margin


def _cwf_prepare_batch(images, resolution, margin):
    """Reflect-pad the downsampled image; the original square remains untouched."""
    small = _cwf_downsample_batch(images, resolution)
    return np.pad(small, ((0, 0), (margin, margin), (margin, margin)), mode="reflect")


class PaddedCWFSource(ImageSource):
    """Serve padded, whitened batches without retaining a full enlarged stack."""

    def __init__(self, images, filters, filter_indices, resolution, margin, pixel_size, noise_stats):
        working_size = resolution + 2 * margin
        super().__init__(L=working_size, n=len(images), dtype=np.float32,
                         pixel_size=pixel_size)
        self.particle_images = images
        self.output_resolution = resolution
        self.margin = margin
        self.noise_mean = noise_stats["mean"]
        self.whitening = PowerFilter(
            ArrayFilter(noise_stats["noise_psd"]), power=-0.5,
            epsilon=noise_stats["whitening_epsilon"],
        )
        # The CTF is evaluated at the same Angstrom/pixel as before padding.
        self.unique_filters = [MultiplicativeFilter(f, self.whitening) for f in filters]
        self.filter_indices = np.asarray(filter_indices, dtype=int)
        self._mutable = False

    def _images(self, indices):
        batch = _cwf_prepare_batch(
            self.particle_images[indices], self.output_resolution, self.margin
        )
        batch -= self.noise_mean
        whitened = Image(batch, pixel_size=self.pixel_size).filter(self.whitening)
        return self.generation_pipeline.forward(whitened, indices)


def estimate_cwf_noise(noise_path, original_size, original_pixel_size,
                       resolution, batch_size=32, max_images=500,
                       noise_pixel_size=None, padding_factor=1.6, seed=0):
    """Estimate a radial noise PSD from whole empty images, in particle intensity units.

    Uses unitary FFTs: mean PSD equals spatial noise variance. The shared
    noise mean is removed from the DC power, then also from particle images.
    Images are never individually normalized to unit variance.
    """
    if batch_size < 1 or not 16 <= resolution <= original_size:
        raise ValueError("Invalid batch size or output resolution")
    margin, working_size = _cwf_padding(resolution, padding_factor)
    with mrcfile.mmap(str(noise_path), mode="r") as mrc:
        stack = mrc.data
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        if stack.ndim != 3 or stack.shape[1:] != (original_size, original_size):
            raise ValueError("Noise and particle images must have the same original box size")
        if noise_pixel_size is None:
            spacing = np.array([float(mrc.voxel_size.x), float(mrc.voxel_size.y)])
        else:
            spacing = np.array([noise_pixel_size, noise_pixel_size], dtype=float)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Set --noise-pixel-size to the known noise sampling in Angstrom/pixel")
        if not np.allclose(spacing, original_pixel_size, rtol=1e-3):
            raise ValueError(f"Noise sampling {spacing} does not match particle sampling {original_pixel_size}")
        count = len(stack) if max_images is None else min(int(max_images), len(stack))
        if count < 2:
            raise ValueError("Use at least two empty images to estimate noise")
        # Use a separate random stream from particle selection. Sort only for I/O;
        # the subset remains uniformly sampled without replacement.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        indices = np.sort(rng.choice(len(stack), count, replace=False))
        print(f"Selected {count} random empty images (seed {seed}).", flush=True)
        power_sum = np.zeros((working_size, working_size), dtype=np.float64)
        pixel_sum = 0.0
        previews = []
        for start in range(0, count, batch_size):
            batch = _cwf_prepare_batch(
                stack[indices[start:start + batch_size]], resolution, margin
            )
            pixel_sum += batch.sum(dtype=np.float64)
            spectrum = np.fft.fftshift(np.fft.fft2(batch, norm="ortho"), axes=(-2, -1))
            power_sum += np.sum(np.abs(spectrum) ** 2, axis=0, dtype=np.float64)
            if len(previews) < 3:
                previews.extend(batch[:3 - len(previews)].copy())

    noise_mean = pixel_sum / (count * working_size**2)
    measured_psd = power_sum / count
    mid = working_size // 2
    # Equivalent to subtracting the SAME scalar noise_mean before each FFT.
    measured_psd[mid, mid] = max(0.0, measured_psd[mid, mid] - working_size**2 * noise_mean**2)
    if not np.isfinite(measured_psd).all() or np.median(measured_psd) <= 0:
        raise ValueError("The empty images do not give a positive finite noise spectrum")
    frequency_pixels = np.fft.fftshift(np.fft.fftfreq(working_size)) * working_size
    yy, xx = np.meshgrid(frequency_pixels, frequency_pixels, indexing="ij")
    rings = np.rint(np.hypot(xx, yy)).astype(int)
    radial_psd = np.bincount(rings.ravel(), weights=measured_psd.ravel())
    radial_psd /= np.maximum(np.bincount(rings.ravel()), 1)
    floor = 0.01 * np.median(measured_psd)
    regularized_radial_psd = np.maximum(radial_psd, floor)
    noise_psd = regularized_radial_psd[rings].astype(np.float32)
    # Match the exact whitening filter used on both particles and CTFs below.
    epsilon = float((100 * np.finfo(np.float32).eps) ** 2)
    whitening = PowerFilter(ArrayFilter(noise_psd), power=-0.5, epsilon=epsilon)
    response = whitening.evaluate_grid(working_size).astype(np.float64)
    whitened_variance = float(np.mean(measured_psd * response**2))
    if not np.isfinite(whitened_variance) or whitened_variance <= 0:
        raise ValueError("Invalid noise variance after whitening")
    pixel_size = float(original_pixel_size * original_size / resolution)
    return {
        "path": str(noise_path), "count": count, "mean": noise_mean,
        "indices": indices, "seed": seed,
        "variance": float(measured_psd.mean()), "noise_psd": noise_psd,
        "measured_psd": measured_psd, "radial_psd": radial_psd,
        "frequency_inv_angstrom": np.arange(len(radial_psd)) / (working_size * pixel_size),
        "pixel_size_angstrom": pixel_size, "whitening_epsilon": epsilon,
        "whitened_variance": whitened_variance,
        "preview": np.asarray(previews, dtype=np.float32)[:, margin:margin + resolution,
                                                        margin:margin + resolution] - noise_mean,
        "working_size": working_size, "crop_margin": margin,
    }


def _cwf_group_ctfs(meta, max_groups):
    """Bound CTF storage by approximating nearby radial CTFs with representatives.

    ASPIRE angularly averages these filters. Cluster mean defocus and half the
    absolute astigmatic difference, separately for each voltage/Cs/contrast.
    Original particle metadata and image order are retained.
    """
    if not isinstance(max_groups, (int, np.integer)) or max_groups < 1:
        raise ValueError("max_ctf_groups must be a positive integer")
    optics_columns = ["rlnVoltage", "rlnSphericalAberration", "rlnAmplitudeContrast"]
    groups = list(meta.groupby(optics_columns, sort=True).indices.values())
    if len(groups) > max_groups:
        raise ValueError("max_ctf_groups is smaller than the number of distinct optics settings")
    filters = []
    assignment = np.empty(len(meta), dtype=int)
    max_errors = np.zeros(2)
    base, extra = divmod(max_groups, len(groups))
    for optics_index, positions in enumerate(groups):
        part = meta.iloc[positions]
        u = part["rlnDefocusU"].to_numpy(dtype=float)
        v = part["rlnDefocusV"].to_numpy(dtype=float)
        features = np.column_stack(((u + v) / 2, np.abs(u - v) / 2))
        allowed = min(base + int(optics_index < extra), len(np.unique(features, axis=0)))
        if allowed == len(positions):
            labels = np.arange(len(positions))
            centers = features
        else:
            model = KMeans(n_clusters=allowed, random_state=0, n_init=3)
            labels = model.fit_predict(features)
            centers = model.cluster_centers_
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            nearest = members[np.argmin(np.sum((features[members] - centers[label])**2, axis=1))]
            row = part.iloc[nearest]
            assignment[positions[members]] = len(filters)
            max_errors = np.maximum(max_errors, np.max(np.abs(features[members] - features[nearest]), axis=0))
            filters.append(CTFFilter(
                voltage=float(row.rlnVoltage), defocus_u=float(row.rlnDefocusU),
                defocus_v=float(row.rlnDefocusV), defocus_ang=np.deg2rad(float(row.rlnDefocusAngle)),
                Cs=float(row.rlnSphericalAberration), alpha=float(row.rlnAmplitudeContrast),
            ))
    return filters, assignment, max_errors


def _cwf_memory_estimate(image_bytes, count, resolution, batch_size, basis, n_filters):
    """Conservative allocation estimate, not a measurement of total process RSS."""
    block_elements = sum(int(rows) * int(cols) for rows, cols in basis.blk_diag_cov_shape)
    one_ctf_bytes = block_elements * np.dtype(np.float32).itemsize
    # Include CTF/operator copies, two output stacks, and batched Fourier work.
    estimated = (image_bytes + 2 * count * resolution**2 * 4
                 + 4 * n_filters * one_ctf_bytes
                 + batch_size * basis.sz[0]**2 * 64 + 512 * 1024**2)
    return estimated / 1024**3, one_ctf_bytes


def denoise_selected_cwf(images, particles, optics, noise_path, resolution=64,
                         batch_size=32, noise_count=500, noise_pixel_size=None,
                         padding_factor=1.6, max_ctf_groups=32,
                         memory_budget_gb=4.0, noise_seed=0):
    """Experimental CTF-aware CWF on the supplied images only; preserve their order."""
    if not isinstance(images, SelectedMRCStack):
        images = np.asarray(images)
    if images.ndim != 3 or images.shape[1] != images.shape[2]:
        raise ValueError("Expected square images with shape (N, L, L)")
    if len(images) != len(particles) or len(images) < 2:
        raise ValueError("Need at least two images and one matching metadata row per image")
    if not 16 <= resolution <= images.shape[-1]:
        raise ValueError("resolution must be between 16 and the original image size")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # Merge optics settings by group ID, retaining the particle/image order.
    meta = particles.reset_index(drop=True).copy()
    optics_by_group = optics.set_index("rlnOpticsGroup")
    if not optics_by_group.index.is_unique:
        raise ValueError("Optics group IDs must be unique")
    for column in ("rlnImagePixelSize", "rlnVoltage", "rlnSphericalAberration",
                   "rlnAmplitudeContrast"):
        if column not in meta:
            meta[column] = meta["rlnOpticsGroup"].map(optics_by_group[column])
    required = ["rlnImagePixelSize", "rlnVoltage", "rlnSphericalAberration",
                "rlnAmplitudeContrast", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle"]
    if not np.isfinite(meta[required].to_numpy(dtype=float)).all():
        raise ValueError("Missing or invalid CTF/optics metadata")
    for column in ("rlnPhaseShift", "rlnCtfBfactor"):
        if column in meta and not np.allclose(meta[column].to_numpy(dtype=float), 0):
            raise ValueError(f"This preview supports zero {column} only")
    pixel_sizes = meta["rlnImagePixelSize"].to_numpy(dtype=float)
    if np.any(pixel_sizes <= 0) or not np.allclose(pixel_sizes, pixel_sizes[0]):
        raise ValueError("This preview requires one positive pixel size for all images")

    if not np.isfinite(memory_budget_gb) or memory_budget_gb <= 0:
        raise ValueError("memory_budget_gb must be positive")
    margin, working_size = _cwf_padding(resolution, padding_factor)
    filters, filter_indices, grouping_errors = _cwf_group_ctfs(meta, max_ctf_groups)
    print(f"Preparing basis for {len(images)} particles and {len(filters)} CTF groups...", flush=True)
    basis = FFBBasis2D(working_size, dtype=np.float32)
    estimated_gb, one_ctf_bytes = _cwf_memory_estimate(
        images.nbytes, len(images), resolution, batch_size, basis, len(filters)
    )
    print(f"Estimated working memory: {estimated_gb:.2f} GiB / {memory_budget_gb:.2f} GiB budget", flush=True)
    print(f"Maximum CTF grouping differences: mean defocus {grouping_errors[0]:.1f} Å; "
          f"half-astigmatism {grouping_errors[1]:.1f} Å", flush=True)
    if estimated_gb > memory_budget_gb:
        raise MemoryError(
            f"CWF estimate {estimated_gb:.2f} GiB exceeds the {memory_budget_gb:.2f} GiB budget. "
            "Reduce --resolution (e.g. 64), --count, or --ctf-groups before rerunning."
        )

    noise_stats = estimate_cwf_noise(
        noise_path, original_size=images.shape[-1], original_pixel_size=pixel_sizes[0],
        resolution=resolution, batch_size=batch_size, max_images=noise_count,
        noise_pixel_size=noise_pixel_size, padding_factor=padding_factor,
        seed=noise_seed,
    )
    pixel_size = noise_stats["pixel_size_angstrom"]
    margin = noise_stats["crop_margin"]
    working_size = noise_stats["working_size"]
    print(f"CWF workspace: {working_size} × {working_size}; output: {resolution} × {resolution}")
    # Keep the full square for comparisons, without allocating an enlarged stack.
    small = np.empty((len(images), resolution, resolution), dtype=np.float32)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        small[start:stop] = _cwf_downsample_batch(images[start:stop], resolution)
    small -= noise_stats["mean"]

    # Generate padded observations lazily and apply matched padded-noise whitening.
    source_white = PaddedCWFSource(
        images, filters, filter_indices, resolution, margin, pixel_size, noise_stats
    )
    noise_psd = noise_stats["noise_psd"]
    noise_variance = noise_stats["whitened_variance"]

    # ASPIRE estimates rotationally invariant mean/covariance in a Fourier-Bessel basis.
    # Its filter projection angularly averages the CTF (astigmatism is approximate).
    denoiser = DenoiserCov2D(
        source_white, basis=basis, var_noise=noise_variance, batch_size=batch_size,
        covar_opt={"shrinker": "frobenius_norm", "max_iter": 100,
                   "rel_tolerance": 1e-6},
    )
    print("Fitting the shared covariance...", flush=True)
    denoiser.build_denoiser()
    print("Denoising selected particles...", flush=True)
    denoised = np.empty_like(small)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        padded_result = denoiser.denoise[start:stop].asnumpy()
        denoised[start:stop] = padded_result[:, margin:margin + resolution,
                                               margin:margin + resolution]
        if stop == len(images) or start // batch_size % 100 == 0:
            print(f"  Denoised {stop}/{len(images)}", flush=True)
    if not np.isfinite(denoised).all():
        raise RuntimeError("CWF produced non-finite output")
    return {
        "input": small, "denoised": denoised,
        "image_names": meta["rlnImageName"].tolist(),
        "pixel_size_angstrom": pixel_size, "noise_psd": noise_psd,
        "whitened_noise_variance": noise_variance,
        "noise_statistics": noise_stats,
        "working_size": working_size, "crop_margin": margin,
        "ctf_group_count": len(filters), "ctf_group_indices": filter_indices,
        "ctf_group_max_errors_angstrom": grouping_errors,
        "estimated_memory_gb": estimated_gb,
    }


def save_previews(result, output_dir, count):
    """Save a bounded comparison figure and a noise-spectrum diagnostic."""
    count = min(count, len(result["denoised"]))
    if count == 0:
        return
    # Limit each page to 10 particles, including when --preview is large.
    for start in range(0, count, 10):
        stop = min(start + 10, count)
        fig, axes = plt.subplots(stop - start, 2, figsize=(8, 3 * (stop - start)), squeeze=False)
        for row, i in enumerate(range(start, stop)):
            number, stack = result["image_names"][i].split("@", 1)
            for col, key in enumerate(("input", "denoised")):
                image = result[key][i]
                low, high = np.percentile(image, [1, 99])
                axes[row, col].imshow(image, cmap="gray", vmin=low, vmax=high)
                title = "Downsampled input" if col == 0 else "CWF estimate"
                axes[row, col].set_title(f"{title}\n{number}@{Path(stack).name}", fontsize=9)
                axes[row, col].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"comparison_{start + 1:04d}.png", dpi=120)
        plt.close(fig)
    stats = result["noise_statistics"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow(stats["preview"][0], cmap="gray")
    axes[0].set_title("Empty noise reference")
    axes[0].axis("off")
    frequency = stats["frequency_inv_angstrom"]
    keep = (frequency > 0) & (frequency <= 0.5 / stats["pixel_size_angstrom"])
    axes[1].semilogy(frequency[keep], stats["radial_psd"][keep])
    axes[1].set(xlabel="Spatial frequency (1/Å)", ylabel="Noise power",
                title=f"Noise PSD: {stats['count']} images")
    fig.tight_layout()
    fig.savefig(output_dir / "noise_reference.png", dpi=120)
    plt.close(fig)


def save_results(result, particles, output_dir, args, elapsed):
    """Write reusable results with original STAR references and run provenance."""
    stack_name = "denoised.mrcs"
    with mrcfile.new(str(output_dir / stack_name), overwrite=False) as mrc:
        mrc.set_data(result["denoised"])
        mrc.set_image_stack()
        mrc.voxel_size = result["pixel_size_angstrom"]
    metadata = particles.copy()
    metadata["cwfImageName"] = [f"{i + 1:06d}@{stack_name}" for i in range(len(metadata))]
    metadata["cwfCTFGroup"] = result["ctf_group_indices"]
    metadata.to_csv(output_dir / "particles.csv", index=False)
    stats = result["noise_statistics"]
    np.savez(output_dir / "noise_statistics.npz", noise_psd=stats["noise_psd"],
             measured_psd=stats["measured_psd"], radial_psd=stats["radial_psd"],
             frequency_inv_angstrom=stats["frequency_inv_angstrom"],
             noise_mean=stats["mean"], noise_variance=stats["variance"],
             noise_indices_zero_based=stats["indices"], noise_seed=stats["seed"])
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    summary = {
        "configuration": configuration, "output_dir": str(output_dir),
        "elapsed_seconds": elapsed, "particle_count": len(particles),
        "output_shape": list(result["denoised"].shape),
        "pixel_size_angstrom": result["pixel_size_angstrom"],
        "working_size": result["working_size"],
        "ctf_group_count": result["ctf_group_count"],
        "ctf_group_max_errors_angstrom": result["ctf_group_max_errors_angstrom"].tolist(),
        "noise_image_count": stats["count"],
        "whitened_noise_variance": result["whitened_noise_variance"],
        "estimated_memory_gb": result["estimated_memory_gb"],
        "assumptions": ["Representative grouped CTFs; angularly averaged astigmatism",
                        "Particle/noise intensity scaling must match",
                        "Reflection padding is an approximate boundary condition",
                        "No particle alignment applied; no ground-truth quality measurement"],
    }
    (output_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_previews(result, output_dir, args.preview)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--star", type=Path, default=DATA_ROOT / "tam_dimer_tetramer" / "J599_split_0_exported.star")
    parser.add_argument("--particles", type=Path, default=DATA_ROOT / "tam_dimer_tetramer" / "batch_0_restacked.mrc")
    parser.add_argument("--noise", type=Path, default=DATA_ROOT / "noise_restacked.mrc")
    parser.add_argument("--stack-reference", help="Original stack path inside rlnImageName, if basename matching is ambiguous")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--selection", choices=("first", "random"), default="random")
    parser.add_argument("--seed", type=int, default=0, help="Nonnegative seed for particle and empty-image sampling; default: 0")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ctf-groups", type=int, default=32)
    parser.add_argument("--memory-budget-gb", type=float, default=4.0)
    parser.add_argument("--padding-factor", type=float, default=1.6)
    parser.add_argument("--noise-count", type=int, default=500, help="Random empty-image sample size; default: 500 (capped at stack size)")
    parser.add_argument("--noise-pixel-size", type=float, default=None, help="Override noise MRC header sampling, in Å/pixel")
    parser.add_argument("--preview", type=int, default=10, help="Number of comparisons to save; 0 disables figures")
    parser.add_argument("--output-dir", type=Path, help="New or empty directory; default: a unique folder in the project-level outputs/ alongside LatentSpace/")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.count < 2 or args.batch_size < 1 or args.threads < 1 or args.ctf_groups < 1:
        parser.error("count must be >= 2; batch-size, threads, and ctf-groups must be positive")
    if args.preview < 0 or (args.noise_count is not None and args.noise_count < 2):
        parser.error("preview must be nonnegative; noise-count must be >= 2")
    if args.resolution < 16 or not np.isfinite(args.memory_budget_gb) or args.memory_budget_gb <= 0:
        parser.error("resolution must be >= 16 and memory-budget-gb must be positive")
    _cwf_padding(args.resolution, args.padding_factor)
    for name in ("star", "particles", "noise"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f"Missing {name} file: {path}")
        setattr(args, name, path)
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            parser.error("output-dir must be new or empty to protect existing results")
    return args


def main():
    args = parse_args()
    logging.getLogger("aspire").setLevel(logging.WARNING)
    started = perf_counter()
    particles, optics, indices = select_particles(
        args.star, args.particles, args.count, args.selection, args.seed, args.stack_reference
    )
    print(f"Selected {len(particles)} particles ({args.selection}, seed {args.seed}); "
          "original stack will be read on demand.", flush=True)
    with mrcfile.mmap(str(args.particles), mode="r") as mapped:
        images = SelectedMRCStack(mapped.data, indices)
        with threadpool_limits(limits=args.threads):
            result = denoise_selected_cwf(
                images, particles, optics, args.noise, resolution=args.resolution,
                batch_size=args.batch_size, noise_count=args.noise_count,
                noise_pixel_size=args.noise_pixel_size, padding_factor=args.padding_factor,
                max_ctf_groups=args.ctf_groups, memory_budget_gb=args.memory_budget_gb,
                noise_seed=args.seed,
            )
    elapsed = perf_counter() - started
    if args.output_dir is None:
        parent = PROJECT_ROOT / "outputs"
        parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="cwf_", dir=parent))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    save_results(result, particles, output_dir, args, elapsed)
    print(f"Completed {len(particles)} particles in {elapsed:.1f} seconds.", flush=True)
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
