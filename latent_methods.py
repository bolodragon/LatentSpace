"""Helpers for expeirment.ipynb: raw loading, CWF, ICA, and paired displays.

Only tam_dimer_tetramer particles and noise_restacked images are used.
CWF depends on ASPIRE; ICA uses scikit-learn. No work is run on import.
"""
from pathlib import Path
import warnings
import logging
import numpy as np
import pandas as pd
import mrcfile
import starfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

# CWF imports are lazy so section 3 can run without ASPIRE installed.
def _import_cwf():
    global FFBBasis2D, Image, ImageSource, CTFFilter, ArrayFilter, PowerFilter, MultiplicativeFilter, DenoiserCov2D, PaddedCWFSource
    from aspire.basis import FFBBasis2D
    from aspire.image import Image
    from aspire.source import ImageSource
    from aspire.operators import CTFFilter, ArrayFilter, PowerFilter, MultiplicativeFilter
    from aspire.denoising import DenoiserCov2D
    PaddedCWFSource = _make_padded_source_class(ImageSource)
    logging.getLogger("aspire").setLevel(logging.WARNING)

def load_raw_dataset(star_path, stack_path, stack_reference, noise_path,
                     particle_count=2000, batch_size=32):
    """Return raw float32 images and matching provenance, without denoising."""
    if particle_count < 2 or batch_size < 1:
        raise ValueError("Need at least two particles and a positive batch size.")
    tables = starfile.read(star_path, always_dict=True)
    particles = tables["particles"]
    refs = particles["rlnImageName"].astype(str).str.split("@", n=1, expand=True)
    if refs.shape[1] != 2 or refs[1].isna().any():
        raise ValueError("Invalid STAR image references; expected index@stack.")
    selected = particles.loc[refs[1] == stack_reference].head(particle_count).copy()
    if len(selected) != particle_count:
        raise ValueError(f"Requested {particle_count} particles; found only {len(selected)}.")
    selected = selected.reset_index(drop=True)
    indices = selected["rlnImageName"].str.split("@", n=1).str[0].astype(np.int64).to_numpy() - 1
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Selected STAR rows contain duplicate particle images.")
    if "rlnImagePixelSize" in selected:
        pixel_sizes = selected["rlnImagePixelSize"].to_numpy(dtype=float)
    else:
        optics = tables["optics"].set_index("rlnOpticsGroup")
        if not optics.index.is_unique:
            raise ValueError("STAR optics group IDs must be unique.")
        pixel_sizes = selected["rlnOpticsGroup"].map(optics["rlnImagePixelSize"]).to_numpy(dtype=float)
    if (not np.isfinite(pixel_sizes).all() or np.any(pixel_sizes <= 0)
            or not np.allclose(pixel_sizes, pixel_sizes[0])):
        raise ValueError("Need one positive STAR pixel size for the selected particles.")

    with mrcfile.mmap(str(stack_path), mode="r") as pmrc, mrcfile.mmap(str(noise_path), mode="r") as nmrc:
        particle_stack = pmrc.data
        noise_stack = nmrc.data
        if noise_stack.ndim == 2:
            noise_stack = noise_stack[None, ...]
        if (particle_stack.ndim != 3 or noise_stack.ndim != 3
                or particle_stack.shape[1:] != noise_stack.shape[1:]):
            raise ValueError("Particle and noise stacks must contain equally sized 2D images.")
        if np.any(indices < 0) or np.any(indices >= len(particle_stack)):
            raise IndexError("STAR reference points outside the particle stack.")
        for name, mrc in (("particle", pmrc), ("noise", nmrc)):
            spacing = np.array([float(mrc.voxel_size.x), float(mrc.voxel_size.y)])
            if not np.allclose(spacing, pixel_sizes[0], rtol=1e-3):
                raise ValueError(f"{name} MRC pixel size {spacing} disagrees with STAR {pixel_sizes[0]}.")
        noise_count = len(noise_stack)
        if noise_count < 1:
            raise ValueError("The noise stack is empty.")
        combined = np.empty((particle_count + noise_count, *particle_stack.shape[1:]), dtype=np.float32)
        for start in range(0, particle_count, batch_size):
            stop = min(start + batch_size, particle_count)
            combined[start:stop] = particle_stack[indices[start:stop]]
        for start in range(0, noise_count, batch_size):
            stop = min(start + batch_size, noise_count)
            combined[particle_count + start:particle_count + stop] = noise_stack[start:stop]
        for start in range(0, len(combined), batch_size):
            if not np.isfinite(combined[start:start + batch_size]).all():
                raise ValueError("Input images contain non-finite values.")

    selected["source"] = "particle"
    selected["stack_index_1based"] = indices + 1
    noise_meta = pd.DataFrame({
        "rlnImageName": [f"{i + 1:06d}@{noise_path.name}" for i in range(noise_count)],
        "source": "noise", "stack_index_1based": np.arange(1, noise_count + 1),
    })
    metadata = pd.concat([selected, noise_meta], ignore_index=True)
    return combined, metadata




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


def _make_padded_source_class(ImageSource):
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
    
    return PaddedCWFSource


def estimate_cwf_noise(noise_path, original_size, original_pixel_size,
                       resolution, batch_size=32, max_images=None,
                       noise_pixel_size=None, padding_factor=1.6):
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
            raise ValueError("Set cwf_noise_pixel_size to the known noise sampling in Angstrom/pixel")
        if not np.allclose(spacing, original_pixel_size, rtol=1e-3):
            raise ValueError(f"Noise sampling {spacing} does not match particle sampling {original_pixel_size}")
        count = len(stack) if max_images is None else min(int(max_images), len(stack))
        if count < 2:
            raise ValueError("Use at least two empty images to estimate noise")
        # Spread a limited selection across the stack; otherwise use every image.
        indices = np.linspace(0, len(stack) - 1, count, dtype=int)
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
                         batch_size=32, noise_count=None, noise_pixel_size=None,
                         padding_factor=1.6, max_ctf_groups=32,
                         memory_budget_gb=4.0, noise_images=None):
    """Experimental CTF-aware CWF on the supplied images only; preserve their order."""
    _import_cwf()
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
        images.nbytes + (0 if noise_images is None else noise_images.nbytes),
        len(images) + (0 if noise_images is None else len(noise_images)),
        resolution, batch_size, basis, len(filters)
    )
    print(f"Estimated working memory: {estimated_gb:.2f} GiB / {memory_budget_gb:.2f} GiB budget", flush=True)
    print(f"Maximum CTF grouping differences: mean defocus {grouping_errors[0]:.1f} Å; "
          f"half-astigmatism {grouping_errors[1]:.1f} Å", flush=True)
    if estimated_gb > memory_budget_gb:
        raise MemoryError(
            f"CWF estimate {estimated_gb:.2f} GiB exceeds the {memory_budget_gb:.2f} GiB budget. "
            "Reduce cwf_resolution (e.g. 64), particle count, or cwf_max_ctf_groups before rerunning."
        )

    noise_stats = estimate_cwf_noise(
        noise_path, original_size=images.shape[-1], original_pixel_size=pixel_sizes[0],
        resolution=resolution, batch_size=batch_size, max_images=noise_count,
        noise_pixel_size=noise_pixel_size, padding_factor=padding_factor,
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
    print("Fitting covariance and denoising in batches...", flush=True)
    denoised = np.empty_like(small)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        padded_result = denoiser.denoise[start:stop].asnumpy()
        denoised[start:stop] = padded_result[:, margin:margin + resolution,
                                               margin:margin + resolution]
    if not np.isfinite(denoised).all():
        raise RuntimeError("CWF produced non-finite output")
    # Apply the already fitted covariance to empty images as a null control.
    # Noise has no CTF metadata: explicitly cycle through selected particle CTFs.
    noise_input = noise_denoised = noise_ctf_indices = None
    if noise_images is not None:
        noise_images = np.asarray(noise_images)
        if noise_images.ndim != 3 or noise_images.shape[1:] != images.shape[1:]:
            raise ValueError("Noise control images must match the raw particle box size")
        noise_ctf_indices = np.resize(filter_indices, len(noise_images))
        noise_source = PaddedCWFSource(
            noise_images, filters, noise_ctf_indices, resolution, margin, pixel_size, noise_stats
        )
        noise_input = np.empty((len(noise_images), resolution, resolution), dtype=np.float32)
        noise_denoised = np.empty_like(noise_input)
        for start in range(0, len(noise_images), batch_size):
            stop = min(start + batch_size, len(noise_images))
            noise_input[start:stop] = _cwf_downsample_batch(noise_images[start:stop], resolution) - noise_stats["mean"]
            coefs = basis.evaluate_t(noise_source.images[start:stop])
            filtered = denoiser.cov2d.get_cwf_coefs(
                coefs, denoiser.cov2d.ctf_basis, noise_ctf_indices[start:stop],
                mean_coef=denoiser.mean_est, covar_coef=denoiser.covar_est,
                noise_var=noise_variance,
            )
            padded = basis.evaluate(filtered).asnumpy()
            noise_denoised[start:stop] = padded[:, margin:margin + resolution, margin:margin + resolution]
        if not np.isfinite(noise_denoised).all():
            raise RuntimeError("CWF noise control produced non-finite output")
    return {
        "noise_input": noise_input, "noise_denoised": noise_denoised,
        "noise_ctf_indices": noise_ctf_indices,
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




def fit_ica_denoiser(images, metadata, n_components=20, seed=0, max_iter=2000,
                     tol=1e-4, batch_size=32, threads=2):
    """Fit directions on 60%, calibrate on 20%, leave remaining 20% untouched."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    images = np.asarray(images)
    if images.ndim != 3 or len(images) != len(metadata):
        raise ValueError("Need one metadata row per image")
    if set(metadata["source"]) != {"particle", "noise"}:
        raise ValueError("Expected particle and noise source labels")
    X = images.reshape(len(images), -1)  # Flattening is a view, not dimension reduction.
    meta = metadata.copy()
    meta["split"] = "test"
    rng = np.random.default_rng(seed)
    for source in ("particle", "noise"):
        rows = rng.permutation(np.flatnonzero(meta["source"].eq(source)))
        nfit, ncal = int(.6 * len(rows)), int(.2 * len(rows))
        if min(nfit, ncal, len(rows) - nfit - ncal) < 2:
            raise ValueError("Need >= 2 images per source in each split")
        meta.iloc[rows[:nfit], meta.columns.get_loc("split")] = "fit"
        meta.iloc[rows[nfit:nfit+ncal], meta.columns.get_loc("split")] = "calibration"
    def rows_for(source=None, split=None):
        mask = np.ones(len(meta), dtype=bool)
        if source is not None:
            mask &= meta["source"].eq(source).to_numpy()
        if split is not None:
            mask &= meta["split"].eq(split).to_numpy()
        return np.flatnonzero(mask)
    fit_rows = rows_for(split="fit")
    if not 2 <= n_components < min(len(fit_rows), X.shape[1]):
        raise ValueError("Invalid component count for the fit matrix")
    noise_rows = rows_for("noise", "fit")
    total = sum(X[noise_rows[i:i+batch_size]].sum(dtype=np.float64)
                for i in range(0, len(noise_rows), batch_size))
    background = float(total / (len(noise_rows) * X.shape[1]))
    with threadpool_limits(limits=threads):
        pca = PCA(n_components=n_components, svd_solver="randomized", iterated_power=5, random_state=seed)
        fit_pixels = X[fit_rows]
        pca.fit(fit_pixels)
        fit_scores = pca.transform(fit_pixels).astype(np.float64)
        del fit_pixels
        if pca.singular_values_[-1] <= 1e-6 * pca.singular_values_[0]:
            raise ValueError("Rank-deficient PCA subspace; reduce n_components")
        ica = FastICA(n_components=n_components, whiten="unit-variance", whiten_solver="svd",
                      max_iter=max_iter, tol=tol, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            ica.fit(fit_scores)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError("ICA did not converge; adjust components/iterations before reconstructing")
        for warning in caught:
            warnings.warn(str(warning.message), warning.category)
        analysis = ica.components_ @ pca.components_
        synthesis = ica.mixing_.T @ pca.components_
        def energy(rows):
            total = np.zeros(n_components, dtype=np.float64)
            for start in range(0, len(rows), batch_size):
                x = X[rows[start:start+batch_size]].astype(np.float64) - background
                z = x @ analysis.T
                total += np.square(z).sum(axis=0)
            return total / len(rows)
        particle_energy = energy(rows_for("particle", "calibration"))
        noise_energy = energy(rows_for("noise", "calibration"))
        weights = np.divide(np.maximum(particle_energy - noise_energy, 0), particle_energy,
                            out=np.zeros_like(particle_energy), where=particle_energy > 0)
    if not np.isfinite(weights).all():
        raise RuntimeError("Non-finite ICA weights")
    return {"analysis": analysis, "synthesis": synthesis, "weights": weights,
            "background": background, "shape": images.shape[1:], "metadata": meta,
            "pca": pca, "ica": ica, "threads": threads,
            "diagnostics": pd.DataFrame({"component": np.arange(1,n_components+1),
                "particle_energy": particle_energy, "noise_energy": noise_energy, "weight": weights})}


def apply_ica_denoiser(images, model, batch_size=32, weights=None):
    """Apply frozen linear ICA shrinkage; output has scalar background removed."""
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != tuple(model["shape"]) or batch_size < 1:
        raise ValueError("Image dimensions or batch size are invalid")
    weights = np.asarray(model["weights"] if weights is None else weights, dtype=float)
    if (weights.shape != (model["analysis"].shape[0],) or not np.isfinite(weights).all()
            or np.any((weights < 0) | (weights > 1))):
        raise ValueError("Need one finite weight in [0, 1] per component")
    output = np.empty(images.shape, dtype=np.float32)
    with threadpool_limits(limits=model["threads"]):
        for start in range(0, len(images), batch_size):
            block = images[start:start+batch_size].reshape(-1, int(np.prod(model["shape"]))).astype(np.float64)
            if not np.isfinite(block).all():
                raise ValueError("Non-finite input pixels")
            z = (block - model["background"]) @ model["analysis"].T
            output[start:start+len(block)] = ((z * weights) @ model["synthesis"]).reshape(-1,*model["shape"])
    return output


def show_raw(particles, noise, particle_names, count=4):
    count = min(count, len(particles), len(noise))
    if count < 1:
        raise ValueError("Need a positive preview count")
    fig, axes = plt.subplots(2, count, figsize=(3*count, 6), squeeze=False)
    for i in range(count):
        for row, image in enumerate((particles[i], noise[i])):
            low, high = np.percentile(image, [1, 99])
            axes[row,i].imshow(image, cmap="gray", vmin=low, vmax=high)
            label = f"Particle {str(particle_names[i]).split('@')[0]}" if row == 0 else f"Noise {i+1}"
            axes[row,i].set_title(label)
            axes[row,i].axis("off")
    fig.suptitle("Raw images — independent contrast for this gallery")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_pairs(particle_input, particle_output, noise_input, noise_output,
               particle_names, method, count=4, particle_indices=None, noise_indices=None,
               input_label="Raw − background", noise_names=None):
    """Four columns, matched image rows and shared contrast within each pair."""
    pi = np.arange(len(particle_input)) if particle_indices is None else np.asarray(particle_indices)
    ni = np.arange(len(noise_input)) if noise_indices is None else np.asarray(noise_indices)
    count = min(count, len(pi), len(ni))
    if count < 1 or particle_input.shape != particle_output.shape or noise_input.shape != noise_output.shape:
        raise ValueError("Need matching input/output stacks and a positive preview count")
    fig, axes = plt.subplots(count, 4, figsize=(12, 3*count), squeeze=False)
    for row, (p,n) in enumerate(zip(pi[:count],ni[:count])):
        for col, raw, processed in ((0, particle_input[p], particle_output[p]),
                                    (2, noise_input[n], noise_output[n])):
            low, high = np.percentile(raw, [1, 99])
            for ax, image in zip(axes[row,col:col+2], (raw,processed)):
                ax.imshow(image,cmap="gray",vmin=low,vmax=high)
                ax.set_xticks([]); ax.set_yticks([])
        axes[row,0].set_ylabel(f"Particle {str(particle_names[p]).split('@')[0]}")
        axes[row,2].set_ylabel(f"Noise {n+1 if noise_names is None else noise_names[n]}")
    for ax,title in zip(axes[0], (f"Particle: {input_label}",f"Particle: {method}",
                                 f"Noise: {input_label}",f"Noise: {method}")):
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{method}: matched images, same contrast within each raw/processed pair")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def fit_noise_pca(images, metadata, dimensions=(20, 50, 100), seed=0,
                  regularization=0.1, batch_size=32, threads=2):
    """Noise-standardized PCA + calibrated generalized covariance directions.

    Test rows only report frozen-direction variance ratios; never tune the models.
    Pixelwise noise standardization is diagonal whitening, not full whitening.
    """
    from scipy.linalg import eigh
    images = np.asarray(images)
    dimensions = tuple(sorted(set(dimensions)))
    if (images.ndim != 3 or len(metadata) != len(images) or batch_size < 1
            or not dimensions or any(not isinstance(k, (int, np.integer)) or k < 2 for k in dimensions)
            or not 0 < regularization <= 1):
        raise ValueError("Invalid images, dimensions, batch size, or regularization")
    if set(metadata['source']) != {'particle', 'noise'}:
        raise ValueError("Need particle and noise source labels")
    X = images.reshape(len(images), -1)
    meta = metadata.copy().reset_index(drop=True)
    meta['split'] = 'test'
    rng = np.random.default_rng(seed)
    for source in ('particle', 'noise'):
        rows = rng.permutation(np.flatnonzero(meta.source.eq(source)))
        a, b = int(.6 * len(rows)), int(.8 * len(rows))
        if min(a, b-a, len(rows)-b) < 3:
            raise ValueError("Too few images in a split")
        meta.loc[rows[:a], 'split'] = 'fit'
        meta.loc[rows[a:b], 'split'] = 'calibration'
    def rows(source=None, split=None):
        mask = np.ones(len(meta), bool)
        if source is not None: mask &= meta.source.eq(source).to_numpy()
        if split is not None: mask &= meta['split'].eq(split).to_numpy()
        return np.flatnonzero(mask)
    def moments(indices):
        total = np.zeros(X.shape[1], np.float64)
        squares = np.zeros_like(total)
        for start in range(0, len(indices), batch_size):
            block = X[indices[start:start+batch_size]].astype(np.float64)
            if not np.isfinite(block).all(): raise ValueError('Non-finite pixels')
            total += block.sum(axis=0)
            squares += np.square(block).sum(axis=0)
        mean = total / len(indices)
        variance = np.maximum((squares - len(indices)*mean**2)/(len(indices)-1), 0)
        return mean, variance
    noise_mean, noise_var = moments(rows('noise', 'fit'))
    particle_mean, particle_var = moments(rows('particle', 'fit'))
    floor = .1 * np.median(noise_var)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('Noise must have positive finite pixel variance')
    scale = np.sqrt(np.maximum(noise_var, floor))
    fit_rows = rows(split='fit')
    if dimensions[-1] >= min(len(fit_rows), X.shape[1], len(rows('noise','calibration'))):
        raise ValueError('Maximum dimension must be below fit size and noise calibration count')
    with threadpool_limits(limits=threads):
        fit_pixels = X[fit_rows].astype(np.float32)
        fit_pixels -= noise_mean.astype(np.float32)
        fit_pixels /= scale.astype(np.float32)
        pca = PCA(n_components=dimensions[-1], svd_solver='randomized',
                  iterated_power=5, random_state=seed).fit(fit_pixels)
        del fit_pixels
        scores = np.empty((len(X), dimensions[-1]), np.float64)
        for start in range(0, len(X), batch_size):
            block = (X[start:start+batch_size].astype(np.float64)-noise_mean)/scale
            # Linear coordinates about the noise mean, without PCA's fit-mean offset.
            scores[start:start+len(block)] = block @ pca.components_.T
        models, summaries = {}, []
        for k in dimensions:
            pc = scores[rows('particle','calibration'), :k]
            nc = scores[rows('noise','calibration'), :k]
            Cp, Cn = np.cov(pc, rowvar=False), np.cov(nc, rowvar=False)
            Cn_reg = (1-regularization)*Cn + regularization*np.trace(Cn)/k*np.eye(k)
            eigenvalues, V = eigh(Cp, Cn_reg)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, V = eigenvalues[order], V[:,order]
            gains = np.maximum(0, 1 - 1/np.maximum(eigenvalues, np.finfo(float).eps))
            # A full inverse, not V.T: generalized eigenvectors are not Euclidean orthonormal.
            inverse = np.linalg.inv(V)
            tp = scores[rows('particle','test'), :k] @ V
            tn = scores[rows('noise','test'), :k] @ V
            test_ratio = tp.var(axis=0, ddof=1) / np.maximum(tn.var(axis=0, ddof=1), np.finfo(float).eps)
            table = pd.DataFrame({'direction':np.arange(1,k+1),
                                  'calibration_ratio_regularized':eigenvalues,
                                  'test_ratio_unregularized':test_ratio,'gain':gains})
            models[k] = {'components':pca.components_[:k], 'directions':V, 'inverse':inverse,
                         'gains':gains, 'particle_mean_scores':pc.mean(axis=0),
                         'noise_mean':noise_mean, 'scale':scale, 'shape':images.shape[1:],
                         'threads':threads, 'diagnostics':table, 'Cp':Cp, 'Cn_regularized':Cn_reg}
            summaries.append({'dimensions':k,'fit_variance_retained':pca.explained_variance_ratio_[:k].sum(),
                              'positive_calibration_gains':int((gains>0).sum()),
                              'top_direction_test_ratio':test_ratio[0],
                              'median_test_ratio':float(np.median(test_ratio))})
    test_particle_mean, _ = moments(rows('particle','test'))
    test_noise_mean, _ = moments(rows('noise','test'))
    return {'models':models,'metadata':meta,'pca':pca,'scores':scores,
            'summary':pd.DataFrame(summaries),
            'mean_difference_fit':(particle_mean-noise_mean).reshape(images.shape[1:]),
            'mean_difference_test':(test_particle_mean-test_noise_mean).reshape(images.shape[1:]),
            'background_summary':pd.DataFrame({
                'source':['particle','noise'], 'fit_mean_intensity':[particle_mean.mean(),noise_mean.mean()],
                'fit_mean_pixel_variance':[particle_var.mean(),noise_var.mean()]})}


def apply_noise_pca(images, model, mode='shrinkage', batch_size=32):
    """Predict signal relative to the fit noise mean; preserve pixel intensity units.

    projection: plain PCA projection; mean: shared calibration particle mean;
    shrinkage: shrink fluctuations toward that mean in noise-referenced directions.
    """
    images = np.asarray(images)
    if images.ndim != 3 or images.shape[1:] != tuple(model['shape']) or batch_size < 1:
        raise ValueError('Invalid image shape or batch size')
    if mode not in ('projection','mean','shrinkage'):
        raise ValueError('Unknown reconstruction mode')
    output = np.empty(images.shape, np.float32)
    with threadpool_limits(limits=model['threads']):
        for start in range(0,len(images),batch_size):
            block=images[start:start+batch_size].reshape(-1,len(model['scale'])).astype(np.float64)
            if not np.isfinite(block).all(): raise ValueError('Non-finite images')
            z=((block-model['noise_mean'])/model['scale']) @ model['components'].T
            if mode=='mean':
                z=np.broadcast_to(model['particle_mean_scores'],z.shape)
            elif mode=='shrinkage':
                t=(z-model['particle_mean_scores']) @ model['directions']
                z=model['particle_mean_scores']+(t*model['gains']) @ model['inverse']
            output[start:start+len(block)]=((z @ model['components'])*model['scale']).reshape(-1,*model['shape'])
    return output
