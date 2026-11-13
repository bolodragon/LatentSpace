#!/usr/bin/env python3
"""Raw cryo-EM -> CWF -> halo treatment, consolidated from the Notion guide.

Source: https://app.notion.com/p/4143c2a71573812ba822c12dc041fe42
All nine original Python modules, including tests and auxiliary workflows, are
retained below with their comments. Module factories keep their namespaces
separate and load scientific dependencies only for the selected command.
No extra project Python files are required.

Place code.py and code.sh in mlls/ alongside batch_0_restacked.mrc and result/.
Examples:
    python code.py cwf --output-dir result/cwf_new
    python code.py halo --input result/cwf_new/particles_cwf.mrc --output-dir halo_new
    python code.py segment --input result/cwf_new/particles_cwf.mrc --output-dir mask_new
    python code.py test
    python code.py test --reference-stack result/cwf_26472411/particles_cwf.mrc
    python code.py pilot --input result/cwf_26472411/particles_cwf.mrc
    python code.py requirements
    bash code.sh all

CWF denoises. Halo correction changes masks only; retained signed CWF pixels
remain unchanged, and outside pixels are multiplied by 0.15. Review masks:
faint features can be lost and noise retained. No molecular identity is inferred.
The class-shapes and lowpass commands are optional original auxiliary workflows,
not additional stages of the raw -> CWF -> halo pipeline.
The recorded environment is available via the requirements command. Matching
historical output also requires the original input data and environment.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import argparse
import os
import sys
import unittest

BASE = Path(__file__).resolve().parent
# Set CRYO_PROJECT_DIR when the data live elsewhere or running an archived copy.
PROJECT_ROOT = Path(os.environ.get("CRYO_PROJECT_DIR", str(BASE))).resolve()
REFERENCE_STACK = None
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("ASPIREDIR", str(PROJECT_ROOT / ".aspire"))

RECORDED_REQUIREMENTS = """aspire==0.14.3
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
clarabel==0.11.1
click==8.5.0
cloudpickle==3.1.2
confuse==2.1.0
contourpy==1.4.0
cvxpy==1.9.3
cycler==0.12.1
finufft==2.5.0
fonttools==4.66.1
gemmi==0.7.5
highspy==1.15.1
idna==3.20
ImageIO==2.38.1
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
lazy-loader==0.6
MarkupSafe==3.0.4
matplotlib==3.11.2
mrcfile==1.5.4
narwhals==2.27.1
networkx==3.7
numpy==2.5.4
osqp==1.1.3
packaging==26.3
pillow==12.3.0
platformdirs==4.13.0
pooch==1.9.0
psutil==7.2.2
pycparser==3.11
pymanopt==2.2.1
pyparsing==3.3.3
python-dateutil==2.9.0.post0
PyWavelets==1.10.0
PyYAML==6.0.3
qdldl==0.1.9.post1
requests==2.34.2
scikit-image==0.26.0
scikit-learn==1.9.1
scipy==1.18.1
scs==3.3.1
setuptools==84.0.0
six==1.17.0
sparsediffpy==0.6.1
threadpoolctl==3.7.0
tifffile==2026.9.20
tqdm==4.70.1
urllib3==2.8.0
"""

# ============================================================================
# Original module: class_shapes.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_class_shapes():
    module_description = """Recover protein views by averaging existing CryoSPARC 2D classes.

    No neural network is needed: images of the same view are aligned and averaged.
    The individual input particles are never overwritten. Half averages diagnose
    repeatability; their inherited alignments do not provide independent validation.
    """

    import argparse
    from concurrent.futures import ThreadPoolExecutor
    import json
    import os
    from pathlib import Path

    BASE = PROJECT_ROOT
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))

    import matplotlib.pyplot as plt
    import mrcfile
    import numpy as np
    from scipy import fft, ndimage, signal


    def metadata_for_stack(path, stack_name, shape):
        """CS rows can be shuffled: blob/idx is the actual MRC slice number."""
        rows = np.load(path, allow_pickle=False)
        names = [Path(p.decode().lstrip(">")).name for p in rows["blob/path"]]
        rows = rows[np.asarray(names) == stack_name]
        if len(np.unique(rows["blob/path"])) != 1:
            raise ValueError("Expected one metadata path matching the input stack.")
        rows = rows[np.argsort(rows["blob/idx"])]
        if not np.array_equal(rows["blob/idx"], np.arange(shape[0])):
            raise ValueError("Metadata must map exactly once to every stack slice.")
        if not np.all(rows["blob/shape"] == shape[1:]):
            raise ValueError("Image dimensions and metadata disagree.")
        apix = float(rows["blob/psize_A"][0])
        if not np.isfinite(apix) or apix <= 0 or not np.allclose(rows["blob/psize_A"], apix):
            raise ValueError("Expected one positive pixel size.")
        return rows, apix


    def resize(image, box):
        """Fourier resampling removes frequencies above the new Nyquist limit."""
        image = np.asarray(image, dtype=np.float32)
        if image.shape != (box, box):
            image = signal.resample(signal.resample(image, box, axis=0), box, axis=1)
        return image.astype(np.float32, copy=False)


    def background_normalize(image):
        yy, xx = np.indices(image.shape)
        radius = np.hypot(yy - (len(image)-1)/2, xx - (len(image)-1)/2)
        background = image[radius > 0.47*len(image)]
        scale = background.std()
        if not np.isfinite(image).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Non-finite or constant particle image.")
        return (image - background.mean()) / scale


    def phase_flip(image, row, apix):
        """CTF sign correction; frequencies are inverse Angstrom, angles radians.

        Formula follows pyem/ctf.py (eval_ctf). Phase flipping avoids dividing by
        CTF zeros, but does not restore amplitudes suppressed by the microscope.
        """
        fy = fft.fftfreq(len(image), d=apix)[:, None]
        fx = fft.rfftfreq(len(image), d=apix)[None, :]
        s2, theta = fx**2 + fy**2, np.arctan2(fy, fx)
        df1, df2 = row["ctf/df1_A"], row["ctf/df2_A"]
        defocus = (df1+df2)/2 + (df1-df2)/2*np.cos(2*(theta-row["ctf/df_angle_rad"]))
        voltage = float(row["ctf/accel_kv"])*1000
        wavelength = 12.2643247 / np.sqrt(voltage*(1 + voltage*0.978466e-6))
        chi = (np.pi*wavelength*defocus*s2
               - np.pi/2*(row["ctf/cs_mm"]*1e7)*wavelength**3*s2**2
               + row["ctf/phase_shift_rad"])
        ac = row["ctf/amp_contrast"]
        ctf = np.sqrt(1-ac**2)*np.sin(chi) + ac*np.cos(chi)
        return fft.irfft2(fft.rfft2(image)*np.where(ctf >= 0, 1, -1), s=image.shape)


    def align(image, row, apix, rotation_sign=-1, shift_sign=1):
        """Invert class -> particle: translate +shift first, then rotate -pose.

        Shift is (x,y) in pixels at alignments2D/psize_A; scipy uses (y,x).
        Source: cryosparc-particle-reposition, cli.py, make_select2d_stamp.
        """
        dy_dx = row["alignments2D/shift"][::-1]*row["alignments2D/psize_A"]/apix
        moved = ndimage.shift(image, shift_sign*dy_dx, order=1, mode="constant", cval=0)
        return ndimage.rotate(moved, rotation_sign*np.degrees(row["alignments2D/pose"]),
                              reshape=False, order=1, mode="constant", cval=0)


    def comparison_vectors(images):
        """Compare shapes inside a broad central disc, with mild noise smoothing."""
        box = images.shape[-1]
        yy, xx = np.indices((box, box))
        mask = np.hypot(yy-(box-1)/2, xx-(box-1)/2) < box*0.40
        values = np.stack([ndimage.gaussian_filter(a, 1.2)[mask] for a in images])
        values -= values.mean(axis=1, keepdims=True)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


    def audit_alignment(source, rows, templates, box, apix, per_class):
        """Check four sign conventions against every template, without assuming order.

        This tests metadata interpretation, not biological validity. We do not search
        arbitrary template rotations, which could conceal a coordinate mistake.
        """
        rng = np.random.default_rng(19)
        classes = np.unique(rows["alignments2D/class"])
        means = np.zeros((4, len(classes), box, box), dtype=np.float64)
        conventions = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        for j, cls in enumerate(classes):
            candidates = np.flatnonzero(rows["alignments2D/class"] == cls)
            indices = rng.choice(candidates, min(per_class, len(candidates)), replace=False)
            for idx in indices:
                row = rows[idx]
                image = background_normalize(resize(source[idx], box))*row["blob/sign"]
                image = phase_flip(image, row, apix)
                for k, (rot, shift) in enumerate(conventions):
                    means[k, j] += align(image, row, apix, rot, shift)/len(indices)
        targets = comparison_vectors(templates)
        scores = []
        for convention, averages in zip(conventions, means):
            correlations = comparison_vectors(averages) @ targets.T
            matches = np.argmax(np.abs(correlations), axis=1)
            best = correlations[np.arange(len(classes)), matches]
            scores.append({"rotation_sign": convention[0], "shift_sign": convention[1],
                           "median_absolute_template_correlation": float(np.median(np.abs(best))),
                           "class_matches": [{"class_id": int(c), "template_index": int(t),
                                              "correlation": float(v)}
                                             for c, t, v in zip(classes, matches, best)]})
        order = np.argsort([s["median_absolute_template_correlation"] for s in scores])[::-1]
        margin = scores[order[0]]["median_absolute_template_correlation"] - scores[order[1]]["median_absolute_template_correlation"]
        return {"conventions": scores, "source_convention_wins": bool(order[0] == 0),
                "winning_margin": float(margin), "particles_per_class": per_class,
                "note": "Template matching is an alignment diagnostic, not independent validation."}


    def save_stack(path, images, apix):
        with mrcfile.new(path, overwrite=False) as output:
            output.set_data(np.asarray(images, dtype=np.float32))
            output.set_image_stack()
            output.voxel_size = apix


    def gallery(images, labels, title, path):
        columns = min(6, len(images))
        fig, axes = plt.subplots(int(np.ceil(len(images)/columns)), columns,
                                 squeeze=False, figsize=(columns*2.4, np.ceil(len(images)/columns)*2.7))
        for ax in axes.flat:
            ax.axis("off")
        for ax, image, label in zip(axes.flat, images, labels):
            lo, hi = np.percentile(image, [1, 99])
            ax.imshow(image, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            ax.set_title(label, fontsize=9)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=150)
        plt.close(fig)


    def main():
        parser = argparse.ArgumentParser(description=module_description)
        parser.add_argument("--input", type=Path, default=BASE/"batch_0_restacked.mrc")
        parser.add_argument("--metadata", type=Path, default=BASE/"result/J599_split_0_exported.cs")
        parser.add_argument("--templates", type=Path, default=BASE/"cryosparc_P543_templates_selected.mrc")
        parser.add_argument("--output-dir", type=Path, default=BASE/"result/class_shapes")
        parser.add_argument("--box", type=int, choices=(64, 128), default=128)
        parser.add_argument("--limit", type=int, help="First N stack slices only, for a smoke test.")
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--pilot-per-class", type=int, default=32)
        args = parser.parse_args()
        if args.workers < 1 or args.pilot_per_class < 1 or (args.limit is not None and args.limit < 1):
            parser.error("Workers, pilot count and limit must be positive.")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with mrcfile.mmap(args.input, mode="r") as stack, mrcfile.open(args.templates) as template_file:
            rows, input_apix = metadata_for_stack(args.metadata, args.input.name, stack.data.shape)
            if not np.allclose([stack.voxel_size.x, stack.voxel_size.y], input_apix):
                raise ValueError("MRC and CS pixel sizes disagree.")
            apix = input_apix*stack.data.shape[-1]/args.box
            if not np.isclose(float(template_file.voxel_size.x)*template_file.data.shape[-1], apix*args.box):
                raise ValueError("Templates and particles have different physical fields of view.")
            templates = np.stack([resize(t, args.box) for t in template_file.data])
            gallery(templates, [f"Template index {i}" for i in range(len(templates))],
                    "Existing CryoSPARC templates; template indices are not class IDs",
                    args.output_dir/"templates.png")
            print("Checking alignment signs against the existing templates...", flush=True)
            audit = audit_alignment(stack.data, rows, templates, args.box, apix, args.pilot_per_class)
            (args.output_dir/"alignment_audit.json").write_text(json.dumps(audit, indent=2))
            for check in audit["conventions"]:
                print({k: v for k, v in check.items() if k != "class_matches"}, flush=True)
            if not audit["source_convention_wins"]:
                raise ValueError("Expected alignment convention did not win; inspect alignment_audit.json before averaging.")
            count = min(args.limit or len(rows), len(rows))
            rows = rows[:count]
            classes = np.unique(rows["alignments2D/class"])
            class_index = {int(c): i for i, c in enumerate(classes)}
            sums = np.zeros((2, 2, len(classes), args.box, args.box), dtype=np.float64)
            counts = np.zeros((2, len(classes)), dtype=int)

            def process(idx):
                row = rows[idx]
                image = background_normalize(resize(stack.data[idx], args.box))*row["blob/sign"]
                corrected = phase_flip(image, row, apix)
                return idx, align(image, row, apix), align(corrected, row, apix)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for start in range(0, count, 256):
                    for idx, raw, corrected in pool.map(process, range(start, min(start+256, count))):
                        cls = class_index[int(rows["alignments2D/class"][idx])]
                        half = int(rows["alignments2D/split"][idx])
                        if half not in (0, 1):
                            raise ValueError("Expected half-set labels 0 or 1.")
                        sums[0, half, cls] += raw
                        sums[1, half, cls] += corrected
                        counts[half, cls] += 1
                    if start % 2048 == 0:
                        print(f"Averaged {min(start+256, count):,}/{count:,} particles", flush=True)
            means = sums.sum(axis=1)/counts.sum(axis=0)[None, :, None, None]
            halves = sums[1]/np.maximum(counts[:, :, None, None], 1)
            for name, images in [("classes_raw_aligned", means[0]), ("classes_phase_flipped", means[1]),
                                 ("half_0", halves[0]), ("half_1", halves[1])]:
                save_stack(args.output_dir/f"{name}.mrc", images, apix)
            labels = [f"Class {c} | n={n:,}" for c, n in zip(classes, counts.sum(axis=0))]
            # The conventional CTF's global sign can differ from CryoSPARC's. Change
            # PNG contrast polarity to match templates; keep saved MRC values intact.
            pilot_correlations = [m["correlation"] for m in audit["conventions"][0]["class_matches"]]
            display_sign = 1 if np.median(pilot_correlations) >= 0 else -1
            gallery(display_sign*means[1], labels, "Phase-corrected, aligned class averages (contrast matched to templates)",
                    args.output_dir/"classes.png")
            gallery(display_sign*means[0], labels, "Aligned class averages without CTF correction", args.output_dir/"classes_raw.png")
            correlations = comparison_vectors(means[1]) @ comparison_vectors(templates).T
            matches = np.argmax(np.abs(correlations), axis=1)
            half_correlation = np.sum(comparison_vectors(halves[0])*comparison_vectors(halves[1]), axis=1)
            report = {"input": str(args.input), "particles": count, "pixel_size_A": apix,
                      "split_field": "alignments2D/split", "alignment_audit": audit,
                      "index_convention": "Class IDs, template indices and output slices are zero-based.",
                      "mrc_intensity": "Background mean subtracted and background standard deviation divided out per particle; blob/sign applied before alignment. Original intensities are not retained.",
                      "mrc_geometry": "Aligned class coordinate system on a Fourier-downsampled pixel grid; class averages are not individual particle slices.",
                      "png_display_sign_relative_to_mrc": display_sign,
                      "ctf_model": "Conventional scalar defocus/astigmatism/voltage/Cs/amplitude-contrast/phase CTF; approximates the exported CryoSPARC spline CTF model.",
                      "limitations": ["Half averages inherit prior class assignments and alignments; not gold-standard validation.",
                                      "Template matches are empirical; class IDs do not imply template order.",
                                      "Phase flipping corrects CTF signs, not amplitudes. Background regions may contain signal."],
                      "classes": [{"output_slice": i, "class_id": int(c), "count": int(counts[:, i].sum()),
                                   "half_counts": counts[:, i].tolist(),
                                   "half_correlation": float(half_correlation[i]) if np.all(counts[:, i] > 0) else None,
                                   "best_template_index": int(matches[i]),
                                   "template_correlation": float(correlations[i, matches[i]])}
                                  for i, c in enumerate(classes)]}
            (args.output_dir/"report.json").write_text(json.dumps(report, indent=2))
            print(f"Saved class shapes, templates, half averages and report to {args.output_dir}", flush=True)
    return SimpleNamespace(metadata_for_stack=metadata_for_stack, resize=resize, background_normalize=background_normalize, phase_flip=phase_flip, align=align, comparison_vectors=comparison_vectors, audit_alignment=audit_alignment, save_stack=save_stack, gallery=gallery, main=main)


# ============================================================================
# Original module: denoise_particles.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_denoise_particles():
    module_description = """Denoise CryoEM particles with a simple low-pass filter for visualization.

    Larger --lowpass values (Angstroms) remove more detail and more noise.
    The output keeps the original particle order, pixel size and intensity sign.
    This does not perform CTF correction, alignment or protein reconstruction.
    """

    import argparse
    import os
    from pathlib import Path

    BASE = PROJECT_ROOT
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))

    import matplotlib.pyplot as plt
    import mrcfile
    import numpy as np
    from scipy import fft

    def load_metadata(path, stack_name, shape):
        """Match metadata to MRC slices using blob/idx, not metadata row order."""
        metadata = np.load(path, allow_pickle=False)
        names = [Path(p.decode().lstrip(">")).name for p in metadata["blob/path"]]
        metadata = metadata[np.array(names) == stack_name]
        if len(np.unique(metadata["blob/path"])) != 1:
            raise ValueError("Metadata must reference exactly one matching stack path.")
        metadata = metadata[np.argsort(metadata["blob/idx"])]
        if not np.array_equal(metadata["blob/idx"], np.arange(shape[0])):
            raise ValueError("Metadata must contain exactly one entry for every MRC slice.")
        if not np.all(metadata["blob/shape"] == shape[1:]):
            raise ValueError("Metadata and MRC image dimensions disagree.")
        pixel_size = float(metadata["blob/psize_A"][0])
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError("Invalid pixel size in metadata.")
        if not np.allclose(metadata["blob/psize_A"], pixel_size):
            raise ValueError("All particles must have the same pixel size.")
        return metadata, pixel_size


    def lowpass(images, pixel_size, resolution, workers=1):
        """Apply a smooth Fourier filter independently to each particle image."""
        images = np.asarray(images, dtype=np.float32)
        if not np.isfinite(images).all():
            raise ValueError("Input contains NaN or infinite values.")
        height, width = images.shape[-2:]

        # Reflect the borders to reduce artifacts from the periodic FFT boundary.
        pad = min(int(np.ceil(2 * resolution / pixel_size)), min(height, width) - 1)
        padded = np.pad(images, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
        fy = fft.fftfreq(padded.shape[-2], d=pixel_size)[:, None]
        fx = fft.rfftfreq(padded.shape[-1], d=pixel_size)[None, :]
        frequency_squared = fx**2 + fy**2

        # Fourth-order Butterworth: half amplitude at frequency 1/resolution.
        transfer = (1 / (1 + (frequency_squared * resolution**2)**4)).astype(np.float32)
        spectrum = fft.rfft2(padded, axes=(-2, -1), workers=workers)
        spectrum *= transfer
        filtered = fft.irfft2(spectrum, s=padded.shape[-2:], axes=(-2, -1), workers=workers)
        return np.ascontiguousarray(filtered[:, pad:pad + height, pad:pad + width])


    def save_preview(images, indices, metadata, pixel_size, resolution, output):
        """Compare raw images with 10, 20, 30 Angstrom filters and the chosen filter."""
        resolutions = sorted({10., 20., 30., resolution})
        versions = [images] + [lowpass(images, pixel_size, r) for r in resolutions]
        titles = ["Raw"] + [f"Low-pass {r:g} Å" for r in resolutions]
        fig, axes = plt.subplots(len(images), len(versions), squeeze=False,
                                 figsize=(2.5 * len(versions), 2.2 * len(images)))
        for row, index in enumerate(indices):
            for col, version in enumerate(versions):
                ax = axes[row, col]
                # CryoSPARC's sign is used for display only, not the saved MRC.
                shown = version[row] * metadata["blob/sign"][index]
                lo, hi = np.percentile(shown, [1, 99])
                ax.imshow(shown, cmap="gray", vmin=lo, vmax=hi, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[col])
                if col == 0:
                    label = f"Slice {index}"
                    if "alignments2D/class" in metadata.dtype.names:
                        label += f"\nClass {metadata['alignments2D/class'][index]}"
                    ax.set_ylabel(label)
        fig.suptitle("Individual particles; zero-based indices\n"
                     "Each panel has its own 1–99% display contrast; metadata sign applied.")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(output, dpi=140)
        plt.close(fig)


    def main():
        parser = argparse.ArgumentParser(description=module_description)
        parser.add_argument("--input", type=Path, default=BASE / "batch_0_restacked.mrc")
        parser.add_argument("--metadata", type=Path, default=BASE / "result/J599_split_0_exported.cs")
        parser.add_argument("--output-dir", type=Path, default=BASE / "result/denoised")
        parser.add_argument("--lowpass", type=float, default=20., help="Filter resolution in Å; default 20.")
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--preview-only", action="store_true", help="Save only a PNG; no large MRC file.")
        parser.add_argument("--limit", type=int, help="Process the first N images for a quick test.")
        args = parser.parse_args()
        if args.batch_size < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
            parser.error("Batch size, workers and limit must be positive.")
        if not np.isfinite(args.lowpass) or args.lowpass <= 0:
            parser.error("Low-pass resolution must be finite and positive.")

        # Memory mapping avoids loading the entire 7.5 GB input into RAM.
        with mrcfile.mmap(args.input, mode="r") as source:
            if source.data.ndim != 3 or min(source.data.shape) < 2:
                raise ValueError("Expected a stack of 2D particle images.")
            metadata, pixel_size = load_metadata(args.metadata, args.input.name, source.data.shape)
            if not np.allclose([source.voxel_size.x, source.voxel_size.y], pixel_size):
                raise ValueError("MRC and metadata pixel sizes disagree.")
            if min(args.lowpass, 10) < 2 * pixel_size:
                raise ValueError("Filter resolution must be at least twice the pixel size.")
            count = min(args.limit or len(source.data), len(source.data))
            print(f"Processing {count:,} particles; pixel size {pixel_size:g} Å", flush=True)

            # Require a fresh directory so earlier results cannot be overwritten.
            args.output_dir.mkdir(parents=True, exist_ok=False)
            indices = np.linspace(0, count - 1, min(8, count), dtype=int)
            images = np.asarray(source.data[indices], dtype=np.float32)
            save_preview(images, indices, metadata, pixel_size, args.lowpass,
                         args.output_dir / "comparison.png")
            if args.preview_only:
                print(f"Preview: {args.output_dir / 'comparison.png'}")
                return

            # A .partial name prevents an interrupted job looking like a completed stack.
            partial = args.output_dir / "particles_denoised.partial.mrc"
            with mrcfile.new_mmap(partial, shape=(count, *source.data.shape[1:]), mrc_mode=2) as dest:
                dest.set_image_stack()
                dest.voxel_size = pixel_size
                dest.header.origin = source.header.origin
                dest.header.label[0] = f"Visualization lowpass {args.lowpass:g} A; original slice order".encode()
                dest.header.nlabl = 1
                total, total_squared = 0., 0.
                minimum, maximum = np.inf, -np.inf
                for first in range(0, count, args.batch_size):
                    last = min(first + args.batch_size, count)
                    filtered = lowpass(source.data[first:last], pixel_size, args.lowpass, args.workers)
                    dest.data[first:last] = filtered
                    # Compute header statistics in batches as well.
                    total += np.sum(filtered, dtype=np.float64)
                    total_squared += np.einsum("ijk,ijk->", filtered, filtered, dtype=np.float64)
                    minimum, maximum = min(minimum, float(filtered.min())), max(maximum, float(filtered.max()))
                    if first == 0 or last == count or first // args.batch_size % 20 == 0:
                        print(f"Saved {last:,}/{count:,} particles", flush=True)
                mean = total / dest.data.size
                dest.header.dmin, dest.header.dmax, dest.header.dmean = minimum, maximum, mean
                dest.header.rms = np.sqrt(max(0, total_squared / dest.data.size - mean**2))
                dest.flush()
            partial.rename(args.output_dir / "particles_denoised.mrc")
            print(f"Results saved in {args.output_dir}")
    return SimpleNamespace(load_metadata=load_metadata, lowpass=lowpass, save_preview=save_preview, main=main)


# ============================================================================
# Original module: particle_segmentation.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_particle_segmentation():
    module_description = """Central-particle segmentation and rotations for the CWF protocol.

    The Gaussian-smoothed, standardized image is used only to select a mask.
    Its reference mean and standard deviation also calibrate the unsmoothed
    pixels before values outside the selected mask are attenuated and rotated.
    """

    import numpy as np
    from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, label, rotate


    SEGMENTATION_THRESHOLDS = (0.1, 0.3, 0.5, 1.1)
    ROTATION_ANGLES = tuple(range(0, 360, 45))
    _CONNECTED = np.ones((3, 3), dtype=bool)


    def _finite_image(image):
        array = np.asarray(image)
        if array.ndim != 2 or not array.size:
            raise ValueError("image must be a nonempty two-dimensional array")
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError("image must contain real numeric values")
        array = np.asarray(array, dtype=np.result_type(array.dtype, np.float32))
        if not np.isfinite(array).all():
            raise ValueError("image must contain only finite values")
        return array


    def _fill_small_holes(mask, growth_region, max_hole_area):
        """Fill enclosed 8-connected background regions up to the area limit."""
        if max_hole_area == 0 or not mask.any():
            return mask
        enclosed = binary_fill_holes(mask, structure=_CONNECTED) & ~mask
        hole_labels, count = label(enclosed, structure=_CONNECTED)
        if not count:
            return mask
        areas = np.bincount(hole_labels.ravel())
        fill = areas <= max_hole_area
        fill[0] = False
        # A hole straddling the growth limit must never enlarge the allowed disk.
        return (mask | fill[hole_labels]) & growth_region


    def segment_center_particle(
        image,
        sigma=1.0,
        seed_radius_fraction=0.18,
        growth_radius_fraction=0.35,
        thresholds=SEGMENTATION_THRESHOLDS,
        seed_threshold=2.0,
        min_area=8,
        max_hole_area=8,
    ):
        """Select the seeded particle at each protocol threshold.

        Radii use the image width and the geometric center ``(shape - 1) / 2``.
        Scores use the population mean and standard deviation of the smoothed
        image inside the inscribed circle. The brightest central pixel must reach
        ``seed_threshold``. Each returned mask is its 8-connected component above
        the requested threshold, restricted to the growth disk, with small holes
        filled. Components smaller than ``min_area`` are rejected before filling.

        ``masks`` has shape ``(len(thresholds), height, width)`` and follows the
        supplied threshold order. ``rows`` holds matching diagnostic records.
        A constant image has zero scores and fails the seed criterion cleanly.
        """
        original = _finite_image(image)
        thresholds = tuple(float(value) for value in thresholds)
        if not thresholds or not np.isfinite(thresholds).all():
            raise ValueError("thresholds must be a nonempty sequence of finite values")
        settings = (sigma, seed_radius_fraction, growth_radius_fraction, seed_threshold)
        if not np.isfinite(settings).all():
            raise ValueError("segmentation settings must be finite")
        if sigma < 0 or not 0 < seed_radius_fraction <= growth_radius_fraction:
            raise ValueError("sigma must be nonnegative and seed/growth radii positive and ordered")
        if int(min_area) != min_area or min_area < 1:
            raise ValueError("min_area must be a positive integer")
        if int(max_hole_area) != max_hole_area or max_hole_area < 0:
            raise ValueError("max_hole_area must be a nonnegative integer")

        height, width = original.shape
        yy, xx = np.ogrid[:height, :width]
        squared_radius = (yy - (height - 1) / 2.0) ** 2 + (xx - (width - 1) / 2.0) ** 2
        reference_region = squared_radius <= (min(height, width) / 2.0) ** 2
        seed_region = squared_radius <= (width * seed_radius_fraction) ** 2
        growth_region = squared_radius <= (width * growth_radius_fraction) ** 2
        growth_boundary = growth_region & ~binary_erosion(growth_region, structure=_CONNECTED)

        smoothed = gaussian_filter(original, sigma=float(sigma))
        reference = smoothed[reference_region]
        mean = reference.mean(dtype=np.float64)
        standard_deviation = reference.std(dtype=np.float64)
        if standard_deviation > 0:
            scores = (smoothed - mean) / standard_deviation
        else:
            scores = np.zeros_like(smoothed)

        seed = None
        seed_score = None
        if seed_region.any():
            seed_index = np.argmax(np.where(seed_region, scores, -np.inf))
            seed = tuple(int(index) for index in np.unravel_index(seed_index, original.shape))
            seed_score = float(scores[seed])

        masks = np.zeros((len(thresholds), height, width), dtype=bool)
        rows = []
        for index, threshold in enumerate(thresholds):
            status = "ok"
            if seed is None:
                status = "no_seed"
            elif seed_score < seed_threshold:
                status = "weak_seed"
            elif seed_score < threshold:
                status = "seed_below_threshold"
            else:
                candidates = (scores >= threshold) & growth_region
                components, _ = label(candidates, structure=_CONNECTED)
                component = components == components[seed]
                if np.count_nonzero(component) < min_area:
                    status = "too_small"
                else:
                    masks[index] = _fill_small_holes(component, growth_region, int(max_hole_area))
            rows.append({
                "threshold": threshold,
                "status": status,
                "area_pixels": int(np.count_nonzero(masks[index])),
                "seed_score": seed_score,
                "touches_growth_boundary": bool(np.any(masks[index] & growth_boundary)),
            })

        return {
            "smoothed": smoothed,
            "scores": scores,
            "reference_mean": float(mean),
            "reference_std": float(standard_deviation),
            "seed_region": seed_region,
            "growth_region": growth_region,
            "seed": seed,
            "masks": masks,
            "rows": rows,
        }


    def standardize_particle_image(image, reference_mean, reference_std):
        """Calibrate unsmoothed pixels using the segmentation reference statistics.

        Removing the per-image reference offset before attenuation prevents a
        bright particle on a negative background becoming darker than its dimmed
        surroundings. A positive scale preserves intensity order and detail;
        signed residuals are retained without clipping or polarity reversal.
        A zero reference standard deviation returns zeros, creating no particle.
        """
        original = _finite_image(image)
        reference_mean = float(reference_mean)
        reference_std = float(reference_std)
        if not np.isfinite(reference_mean) or not np.isfinite(reference_std):
            raise ValueError("reference mean and standard deviation must be finite")
        if reference_std < 0:
            raise ValueError("reference standard deviation must be nonnegative")
        if reference_std == 0:
            return np.zeros(original.shape, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            calibrated = np.asarray(
                (original.astype(np.float64) - reference_mean) / reference_std,
                dtype=np.float32,
            )
        if not np.isfinite(calibrated).all():
            raise ValueError("standardized image must contain only finite float32 values")
        return calibrated


    def mask_iou(first, second):
        """Return intersection / union; two empty masks have undefined (NaN) IoU."""
        first = np.asarray(first, dtype=bool)
        second = np.asarray(second, dtype=bool)
        if first.shape != second.shape:
            raise ValueError("masks must have the same shape")
        union = np.count_nonzero(first | second)
        return float(np.count_nonzero(first & second) / union) if union else float("nan")


    def oriented_particle_vectors(image, mask, angles=ROTATION_ANGLES, outside_factor=0.15):
        """Return the dimmed original and its ordered image orientations.

        The second return value is a stack ``(angle, height, width)``. Flattening
        all three axes gives the grouped features for one source image. Positive
        angles rotate counterclockwise around the geometric image center, using
        bilinear interpolation and zero fill without changing the image shape.
        Empty masks dim the entire original image and still produce every view.
        """
        original = _finite_image(image)
        mask = np.asarray(mask)
        if mask.shape != original.shape:
            raise ValueError("mask must have the same shape as image")
        if not np.isfinite(mask).all():
            raise ValueError("mask must contain only finite values")
        mask = mask.astype(bool, copy=False)
        angles = tuple(float(angle) for angle in angles)
        if not angles or not np.isfinite(angles).all():
            raise ValueError("angles must be a nonempty sequence of finite values")
        if not np.isfinite(outside_factor) or not 0 <= outside_factor <= 1:
            raise ValueError("outside_factor must be finite and between zero and one")

        dimmed = original.copy()
        dimmed[~mask] *= outside_factor
        orientations = np.empty((len(angles),) + original.shape, dtype=dimmed.dtype)
        for index, angle in enumerate(angles):
            if angle % 360 == 0:
                orientations[index] = dimmed
            else:
                rotate(
                    dimmed, angle, reshape=False, output=orientations[index],
                    order=1, mode="constant", cval=0.0, prefilter=False,
                )
        return dimmed, orientations
    return SimpleNamespace(_finite_image=_finite_image, _fill_small_holes=_fill_small_holes, segment_center_particle=segment_center_particle, standardize_particle_image=standardize_particle_image, mask_iou=mask_iou, oriented_particle_vectors=oriented_particle_vectors, ROTATION_ANGLES=ROTATION_ANGLES, SEGMENTATION_THRESHOLDS=SEGMENTATION_THRESHOLDS)


# ============================================================================
# Original module: halo_segmentation.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_halo_segmentation():
    module_description = """Remove circular halo from mask scores, preserving the strong central core.

    Original CWF pixels are never background-subtracted. The radial model is
    used only to prevent seed-and-grow segmentation following a circular halo.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label, binary_erosion, binary_dilation
    original_segmentation = _load_particle_segmentation().segment_center_particle


    def segment_center_particle(image, halo_protect_pixels=2., halo_quantile=.75, neck_radius=1, **settings):
        result = original_segmentation(image, **settings)
        masks = result['masks'].copy()
        core_index = int(np.argmax([row['threshold'] for row in result['rows']]))
        # A complex can have multiple bright lobes separated at the high threshold.
        # Protect ALL sufficiently large central high-threshold components already
        # connected in the permissive mask, rather than just the seed's bright lobe.
        strong, count = label((result['scores'] >= result['rows'][core_index]['threshold'])
                              & result['growth_region'], structure=np.ones((3, 3)))
        areas = np.bincount(strong.ravel())
        central_labels = np.unique(strong[result['seed_region']])
        central_labels = central_labels[(central_labels != 0)
                                        & (areas[central_labels] >= settings.get('min_area', 8))]
        weakest_index = int(np.argmin([row['threshold'] for row in result['rows']]))
        core = (masks[core_index] | np.isin(strong, central_labels)) & masks[weakest_index]
        result['original_masks'] = masks.copy()
        result['core_mask'] = core.copy()
        result['halo_removed'] = np.zeros(len(masks), dtype=int)
        if not core.any():
            result['halo_model_scores'] = np.zeros_like(result['scores'])
            return result
        y, x = np.indices(image.shape)
        bins = np.hypot(y-(image.shape[0]-1)/2, x-(image.shape[1]-1)/2).astype(int)
        profile = np.array([np.quantile(result['scores'][bins == r], halo_quantile) for r in range(bins.max()+1)])
        # Suppress only the above-background, circularly symmetric part. Negative
        # annuli are not inverted into positive signal. Keep the original scale.
        profile = np.maximum(gaussian_filter1d(profile, .8), 0)
        # Keep the complete strong core and two pixels around it untouched.
        distance = distance_transform_edt(~core)
        fade = np.clip(distance-halo_protect_pixels, 0, 1)
        halo = profile[bins] * fade
        scores = result['scores'] - halo
        result['halo_model_scores'] = halo
        for j, record in enumerate(result['rows']):
            if not masks[j].any(): continue
            candidate = masks[j] & (scores >= record['threshold'])
            # Hole-filled pixels in the protected original core remain protected.
            candidate |= core & masks[j]
            components, _ = label(candidate, structure=np.ones((3, 3)))
            anchors = np.unique(components[core & candidate]); anchors = anchors[anchors != 0]
            mask = np.isin(components, anchors) & candidate
            if neck_radius:
                # Thin connections can join a protein to an outer arc. Keep thick
                # components anchored in the strong core, then restore their edge.
                eroded = binary_erosion(mask, structure=np.ones((3, 3)), iterations=neck_radius)
                parts, _ = label(eroded, structure=np.ones((3, 3)))
                anchors = np.unique(parts[core]); anchors = anchors[anchors != 0]
                kept = np.isin(parts, anchors) if len(anchors) else np.zeros_like(mask)
                restored = binary_dilation(kept, structure=np.ones((3, 3)), iterations=neck_radius)
                mask &= restored | core
            result['masks'][j] = mask
            result['halo_removed'][j] = int(masks[j].sum()-mask.sum())
            record['area_pixels'] = int(mask.sum())
            # Preserve the original boundary definition.
            boundary = result['growth_region'] & ~binary_erosion(result['growth_region'], structure=np.ones((3, 3)))
            record['touches_growth_boundary'] = bool(np.any(mask & boundary))
        return result
    return SimpleNamespace(segment_center_particle=segment_center_particle)


# ============================================================================
# Original module: cwf_denoise.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_cwf_denoise():
    module_description = """Estimate individual protein outlines with CTF-aware covariance Wiener filtering.

    ASPIRE learns a rotational covariance from the noisy particles, without clean
    training targets. This is a coarse visualization estimate, not a 3D structure.
    Run class_shapes.py separately to see the much clearer averages of shared views.
    """

    import argparse
    import json
    import os
    from pathlib import Path
    import time

    BASE = PROJECT_ROOT
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".matplotlib"))
    os.environ.setdefault("ASPIREDIR", str(BASE / ".aspire"))

    import matplotlib.pyplot as plt
    import mrcfile
    import numpy as np
    from scipy import fft, signal
    from aspire.basis import FFBBasis2D
    from aspire.denoising import DenoiserCov2D
    from aspire.image import Image
    from aspire.noise import AnisotropicNoiseEstimator, WhiteNoiseEstimator
    from aspire.operators import ArrayFilter, RadialCTFFilter
    from aspire.source import ArrayImageSource
    metadata_for_stack = _load_class_shapes().metadata_for_stack
    lowpass = _load_denoise_particles().lowpass


    def prepare(stack, rows, box, apix, batch_size):
        """Downsample, subtract background, scale noise, then recenter each image.

        Fourier translation avoids spatial interpolation blur; no rotation or
        template is applied. The +shift convention was checked against J599 templates.
        """
        count = len(rows)
        images = np.empty((count, box, box), np.float32)
        scales = np.empty(count, np.float32)
        yy, xx = np.indices((box, box))
        background = np.hypot(yy-box/2, xx-box/2) >= box/2
        fy, fx = fft.fftfreq(box)[:, None], fft.fftfreq(box)[None, :]
        for start in range(0, count, batch_size):
            stop = min(start+batch_size, count)
            data = np.asarray(stack[start:stop], dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError("Non-finite input pixels.")
            data = signal.resample(signal.resample(data, box, axis=-1), box, axis=-2)
            noise = data[:, background]
            scale = noise.std(axis=1)
            if np.any(scale <= 0):
                raise ValueError("Constant image background.")
            data = (data-noise.mean(axis=1)[:, None, None])/scale[:, None, None]
            data *= rows["blob/sign"][start:stop, None, None]
            shift = (rows["alignments2D/shift"][start:stop]
                     * rows["alignments2D/psize_A"][start:stop, None]/apix)
            phase = np.exp(-2j*np.pi*(shift[:, 0, None, None]*fx
                                      + shift[:, 1, None, None]*fy))
            images[start:stop] = fft.ifft2(fft.fft2(data)*phase).real
            scales[start:stop] = scale
            if start % (batch_size*16) == 0:
                print(f"Prepared {stop:,}/{count:,} particles", flush=True)
        return images, scales


    def ctf_groups(rows, width):
        """Bin defocus for affordable CWF; quantify the radial CTF approximation."""
        for key in ("ctf/accel_kv", "ctf/cs_mm", "ctf/amp_contrast"):
            if not np.allclose(rows[key], rows[key][0]):
                raise ValueError(f"This simple script requires a constant {key}.")
        if np.any(rows["ctf/phase_shift_rad"] != 0):
            raise ValueError("Nonzero phase-plate shifts need a different CTF model.")
        defocus = (rows["ctf/df1_A"]+rows["ctf/df2_A"])/2
        centers, ids = np.unique(np.round(defocus/width)*width, return_inverse=True)
        filters = [RadialCTFFilter(voltage=float(rows["ctf/accel_kv"][0]),
                                  defocus=float(d), Cs=float(rows["ctf/cs_mm"][0]),
                                  alpha=float(rows["ctf/amp_contrast"][0])) for d in centers]
        # Error from grouping plus the largest direction-dependent defocus change.
        error = abs(defocus-centers[ids])+abs(rows["ctf/df1_A"]-rows["ctf/df2_A"])/2
        voltage = float(rows["ctf/accel_kv"][0])*1000
        wavelength = 12.2643247/np.sqrt(voltage*(1+voltage*0.978466e-6))
        return filters, ids, centers, np.pi*wavelength*error


    def radial_noise(source, training):
        """Estimate the corner PSD and average rings for the rotational CWF model."""
        estimator = AnisotropicNoiseEstimator(source[training[:2048]], bg_radius=1.0)
        psd = estimator.filter.evaluate_grid(source.L).astype(np.float32)
        yy, xx = np.indices(psd.shape)
        rings = np.round(np.hypot(yy-source.L//2, xx-source.L//2)).astype(int)
        spectrum = np.bincount(rings.ravel(), weights=psd.ravel())/np.bincount(rings.ravel())
        radial = np.maximum(spectrum[rings], np.median(psd)*0.01).astype(np.float32)
        usable = (rings > 2) & (rings < source.L/2-2)
        anisotropy = float(np.sqrt(np.mean((psd[usable]/radial[usable]-1)**2)))
        return radial, anisotropy


    def estimate(model, images, filter_ids, ctf_basis):
        """Apply a fitted covariance to any particles, including held-out images."""
        coefficients = model.basis.evaluate_t(images)
        restored = model.cov2d.get_cwf_coefs(
            coefficients, ctf_basis, filter_ids,
            mean_coef=model.mean_est, covar_coef=model.covar_est,
            noise_var=model.var_noise)
        return model.basis.evaluate(restored).asnumpy()


    def comparison(versions, titles, indices, path, title):
        fig, axes = plt.subplots(len(indices), len(versions), squeeze=False,
                                 figsize=(2.8*len(versions), 2.2*len(indices)))
        for r, index in enumerate(indices):
            for c, images in enumerate(versions):
                lo, hi = np.percentile(images[r], [1, 99])
                axes[r, c].imshow(images[r], cmap="gray", origin="lower", vmin=lo, vmax=hi)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(titles[c])
                if c == 0:
                    axes[r, c].set_ylabel(f"Slice {index}")
        fig.suptitle(title+"\nEach panel has its own 1–99% display contrast.")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=140)
        plt.close(fig)


    def main():
        parser = argparse.ArgumentParser(description=module_description)
        parser.add_argument("--input", type=Path, default=BASE/"batch_0_restacked.mrc")
        parser.add_argument("--metadata", type=Path, default=BASE/"result/J599_split_0_exported.cs")
        parser.add_argument("--output-dir", type=Path, default=BASE/"result/cwf")
        parser.add_argument("--box", type=int, choices=(32, 64, 96), default=64)
        parser.add_argument("--defocus-bin", type=float, default=500., help="Defocus grouping width in Angstrom.")
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--limit", type=int, help="First N particles for a small trial.")
        parser.add_argument("--preview-only", action="store_true", help="Fit model and write previews, skip the full output stack.")
        args = parser.parse_args()
        if args.batch_size < 1 or args.defocus_bin <= 0 or not np.isfinite(args.defocus_bin):
            parser.error("Batch size and defocus bin must be positive.")
        if args.limit is not None and args.limit < 128:
            parser.error("Use at least 128 images for a smoke test; thousands for meaningful results.")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        with mrcfile.mmap(args.input) as stack:
            rows, original_apix = metadata_for_stack(args.metadata, args.input.name, stack.data.shape)
            if not np.allclose([stack.voxel_size.x, stack.voxel_size.y], original_apix):
                raise ValueError("MRC and CS pixel sizes disagree.")
            if args.box > stack.data.shape[-1]:
                raise ValueError("Output box must not exceed input dimensions.")
            rows = rows[:args.limit] if args.limit else rows
            apix = original_apix*stack.data.shape[-1]/args.box
            data, scales = prepare(stack.data, rows, args.box, apix, args.batch_size)
        count = len(data)
        rng = np.random.default_rng(71)
        permutation = rng.permutation(count)
        heldout = permutation[:min(128, count//5)]
        # Include the same eight examples as the earlier neural run when available.
        previous = BASE/"result/neural_26469449/heldout_indices.npy"
        preview = np.load(previous) if previous.exists() else heldout[:8]
        preview = preview[preview < count]
        if len(preview) < 8:
            preview = heldout[:8]
        heldout = np.unique(np.concatenate([heldout, preview]))
        training = np.setdiff1d(permutation, heldout)
        rng.shuffle(training)
        np.save(args.output_dir/"covariance_holdout_indices.npy", heldout)
        np.save(args.output_dir/"preview_indices.npy", preview)
        np.save(args.output_dir/"input_background_scales.npy", scales)
        filters, filter_ids, defocus, phase_bound = ctf_groups(rows, args.defocus_bin)
        source = ArrayImageSource(data, pixel_size=apix)
        source.unique_filters, source.filter_indices = filters, filter_ids
        psd, anisotropy = radial_noise(source, training)
        np.save(args.output_dir/"noise_psd.npy", psd)
        white = source.whiten(ArrayFilter(psd)).cache(batch_size=args.batch_size)
        noise_var = WhiteNoiseEstimator(white[training[:2048]], bg_radius=1.0).estimate()
        print(f"CWF: {len(training):,} fitting / {len(heldout)} held out; "
              f"{len(filters)} CTF groups; white noise variance {noise_var:.4f}", flush=True)
        basis = FFBBasis2D((args.box, args.box), dtype=np.float32)
        fit_source = white[training]
        model = DenoiserCov2D(fit_source, basis, var_noise=noise_var,
                             batch_size=args.batch_size,
                             covar_opt={"max_iter": 100, "rel_tolerance": 1e-5})
        model.build_denoiser()
        # IndexedSource compacts its filter IDs. Undo that mapping for inference,
        # including a defocus bin which happens to have no training particles.
        ctf_basis = [None]*len(filters)
        for original_id, operator in zip(np.unique(filter_ids[training]), model.cov2d.ctf_basis):
            ctf_basis[original_id] = operator
        for k, operator in enumerate(ctf_basis):
            if operator is None:
                ctf_basis[k] = basis.filter_to_basis_mat(white.unique_filters[k], pixel_size=apix)
        restored = estimate(model, white.images[preview], filter_ids[preview], ctf_basis)
        raw = data[preview]
        phase_flipped = np.stack([Image(raw[i], pixel_size=apix).filter(
            filters[filter_ids[j]].sign).asnumpy()[0] for i, j in enumerate(preview)])
        display_resolution = max(25., 2*apix)
        filter_label = f"{display_resolution:g} Å"
        comparison([lowpass(raw, apix, display_resolution), lowpass(phase_flipped, apix, display_resolution),
                    restored, lowpass(restored, apix, display_resolution)],
                   [f"Input + {filter_label} filter", f"CTF phase flip + {filter_label}",
                    "CWF estimate", f"CWF + {filter_label}"],
                   preview, args.output_dir/"comparison.png",
                   "Individual particles excluded from covariance fitting; centered using J599 shifts")
        with mrcfile.new(args.output_dir/"heldout_denoised.mrc") as out:
            out.set_data(restored.astype(np.float32))
            out.set_image_stack()
            out.voxel_size = apix
        # A nonzero prior mean can leave structure even when the input is pure noise.
        null = rng.normal(0, np.sqrt(noise_var), size=(4, args.box, args.box)).astype(np.float32)
        null_result = estimate(model, Image(null), filter_ids[preview[:4]], ctf_basis)
        comparison([null, null_result], ["Synthetic white noise", "CWF estimate"], np.arange(4),
                   args.output_dir/"noise_control.png", "No protein signal was added to these controls")
        # Match every held-out particle's CTF, so validation can distinguish the
        # response to measured data from structure produced without protein signal.
        control_rng = np.random.default_rng(1729)
        control_shape = (len(heldout), args.box, args.box)
        control_inputs = {
            "noise_only": control_rng.normal(0, np.sqrt(noise_var), control_shape).astype(np.float32),
            "zero_input": np.zeros(control_shape, dtype=np.float32),
        }
        np.save(args.output_dir/"control_indices.npy", heldout)
        control_rms = {}
        for name, values in control_inputs.items():
            control = estimate(model, Image(values), filter_ids[heldout], ctf_basis)
            if not np.isfinite(control).all():
                raise ValueError(f"Non-finite {name} control.")
            with mrcfile.new(args.output_dir/f"{name}_cwf.mrc") as out:
                out.set_data(control.astype(np.float32))
                out.set_image_stack()
                out.voxel_size = apix
            control_rms[name] = float(np.sqrt(np.mean(control**2)))
        if not args.preview_only:
            partial = args.output_dir/"particles_cwf.partial.mrc"
            with mrcfile.new_mmap(partial, shape=data.shape, mrc_mode=2) as out:
                out.set_image_stack()
                out.voxel_size = apix
                out.header.label[0] = b"CWF visualization; centered; CS sign applied; background-normalized"
                out.header.nlabl = 1
                for first in range(0, count, args.batch_size):
                    last = min(first+args.batch_size, count)
                    result = estimate(model, white.images[first:last], filter_ids[first:last], ctf_basis)
                    if not np.isfinite(result).all():
                        raise ValueError("Non-finite Wiener estimate.")
                    out.data[first:last] = result
                    if first % (16*args.batch_size) == 0:
                        print(f"Wrote {last:,}/{count:,} estimates", flush=True)
                out.reset_header_stats()
                out.flush()
            partial.rename(args.output_dir/"particles_cwf.mrc")
        report = {"method": "ASPIRE 0.14.3 FFBBasis2D covariance Wiener filtering",
                  "input": str(args.input.resolve()), "metadata": str(args.metadata.resolve()),
                  "particles": count, "fit_particles": len(training), "heldout_particles": len(heldout),
                  "box": args.box, "pixel_size_A": apix, "defocus_bins_A": defocus.tolist(),
                  "preview_lowpass_A": display_resolution,
                  "noise_variance_after_whitening": float(noise_var),
                  "noise_psd_angular_relative_rms": anisotropy,
                  "radial_ctf_phase_error_bound_at_20A_p50_p95_max": np.percentile(phase_bound/20**2, [50, 95, 100]).tolist(),
                  "radial_ctf_phase_error_bound_at_Nyquist_p50_p95_max": np.percentile(phase_bound/(2*apix)**2, [50, 95, 100]).tolist(),
                  "null_output_rms": float(np.sqrt(np.mean(null_result**2))),
                  "matched_control_particles": len(heldout),
                  "matched_control_indices": "control_indices.npy",
                  "matched_control_output_rms": control_rms,
                  "particle_preview_output_rms": float(np.sqrt(np.mean(restored**2))),
                  "output_coordinates": "Original slice order, Fourier-recentered using +CS 2D shifts; no in-plane rotation.",
                  "output_intensity": "Background normalized, CS blob/sign applied to inputs; ASPIRE CTF convention; no absolute density units.",
                  "limitations": ["Corner background and stationary radial noise are assumptions, not proven by centering.",
                                  "Grouped radial CTF approximates astigmatism and the exported spline CTF.",
                                  "Existing J599 shifts use previously fitted references; held-out covariance does not make alignment independent.",
                                  "Wiener estimates use a learned prior; visible features are not independent proof of protein structure.",
                                  "The noise stack generated from averages is not used as acquisition noise."],
                  "elapsed_seconds": time.monotonic()-started}
        (args.output_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Finished: {args.output_dir}", flush=True)
    return SimpleNamespace(prepare=prepare, ctf_groups=ctf_groups, radial_noise=radial_noise, estimate=estimate, comparison=comparison, main=main)


# ============================================================================
# Original module: notion_postprocess.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_notion_postprocess():
    module_description = """Run Notion sections 3.1–3.4 on the local individual CWF MRC stack.

    Simple adaptation: replace checkpoint selection with an explicit input MRC,
    write vectors to disk, and retain zero-based original source indices.
    """
    import argparse
    import csv
    import hashlib
    import json
    import logging
    from pathlib import Path
    import shutil
    import sys
    from collections import Counter

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import mrcfile
    import numpy as np

    PROJECT = PROJECT_ROOT
    segment_center_particle = _load_particle_segmentation().segment_center_particle
    mask_iou = _load_particle_segmentation().mask_iou
    oriented_particle_vectors = _load_particle_segmentation().oriented_particle_vectors

    SETTINGS = dict(sigma=1., seed_radius_fraction=.18, growth_radius_fraction=.35,
                    thresholds=[.1, .3, .5, 1.1], seed_threshold=2., min_area=8,
                    max_hole_area=8)
    ANGLES = np.arange(0, 360, 45, dtype=float)
    OUTSIDE_FACTOR = .15
    REFERENCES = [17325, 21949, 16637, 39304, 21528]
    PAGE = 'https://app.notion.com/p/3fa3c2a71573804aaed3cb6ad28eeaf8'


    def segmentation_plot(images, ids, out, filename):
        """Original, smooth/search, mask, contour, and four-threshold comparison."""
        colors = plt.cm.viridis(np.linspace(.15, .9, len(SETTINGS['thresholds'])))
        fig, axes = plt.subplots(len(ids), 5, figsize=(17, 3*len(ids)), squeeze=False)
        for row, index in enumerate(ids):
            image = images[index]
            result = segment_center_particle(image, **SETTINGS)
            masks = result['masks']; chosen = masks[0]
            # Per-particle contrast shared across that row. Never change data.
            low, high = np.percentile(image, [1, 99.5])
            for col in [0, 1, 3, 4]:
                axes[row, col].imshow(result['smoothed'] if col == 1 else image,
                                      cmap='gray', vmin=low, vmax=high)
            for name, color in [('seed_region', 'cyan'), ('growth_region', 'yellow')]:
                axes[row, 1].contour(result[name], levels=[.5], colors=[color], linewidths=.8)
            sy, sx = result['seed']; axes[row, 1].plot(sx, sy, 'r+', markersize=7)
            axes[row, 2].imshow(chosen, cmap='gray', vmin=0, vmax=1)
            if chosen.any(): axes[row, 3].contour(chosen, levels=[.5], colors=['lime'], linewidths=1)
            else: axes[row, 3].set_xlabel('Empty mask')
            overlaps = []
            for threshold, mask, color in zip(SETTINGS['thresholds'], masks, colors):
                if mask.any(): axes[row, 4].contour(mask, levels=[.5], colors=[color], linewidths=.9)
                overlaps.append(f'{threshold:g}: {mask_iou(chosen, mask):.2f}')
            axes[row, 4].set_xlabel('IoU ' + '; '.join(overlaps), fontsize=7)
            axes[row, 0].set_ylabel(f'Source {index}')
            for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
        for ax, title in zip(axes[0], ['Original CWF', 'Smooth + search regions', 'Weakest mask (0.1)',
                                      'Mask on original', 'All four thresholds']): ax.set_title(title, fontsize=10)
        axes[0, 4].legend(handles=[Line2D([0], [0], color=c, label=f'{t:g}')
                                  for t, c in zip(SETTINGS['thresholds'], colors)], fontsize=7)
        fig.suptitle('Notion central segmentation: inspect masks and threshold sensitivity')
        fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(out/filename, dpi=130); plt.close(fig)


    def orientation_plot(images, masks, ids, out):
        fig, axes = plt.subplots(len(ids), 10, figsize=(23, 2.6*len(ids)), squeeze=False)
        for row, index in enumerate(ids):
            image = images[index]
            dimmed, views = oriented_particle_vectors(image, masks[index, 0], ANGLES, OUTSIDE_FACTOR)
            low, high = np.percentile(image, [1, 99.5])
            for ax, panel in zip(axes[row], [image, dimmed, *views]):
                ax.imshow(panel, cmap='gray', vmin=low, vmax=high); ax.set_xticks([]); ax.set_yticks([])
            axes[row, 0].set_ylabel(f'Source {index}')
        for ax, title in zip(axes[0], ['Original CWF', 'Outside ×0.15', *[f'{a:g}°' for a in ANGLES]]):
            ax.set_title(title, fontsize=9)
        fig.suptitle('Eight counterclockwise orientations per original particle; shared contrast within each row')
        fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(out/'03_orientation_preview.png', dpi=125); plt.close(fig)


    def main():
        nonlocal segment_center_particle
        parser = argparse.ArgumentParser(description=module_description)
        parser.add_argument('--input', type=Path, default=PROJECT/'result/cwf_26472411/particles_cwf.mrc')
        parser.add_argument('--output-dir', type=Path, required=True)
        parser.add_argument('--limit', type=int, help='Optional first N particles for a trial')
        parser.add_argument('--ignore-halo', action='store_true', help='Reject radial halo and narrow connections outside protected bright cores')
        args = parser.parse_args()
        if args.ignore_halo:
            segment_center_particle = _load_halo_segmentation().segment_center_particle
        out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[logging.FileHandler(out/'run.log'), logging.StreamHandler()])
        log = logging.getLogger('notion_postprocess')
        snapshot = out/'source_code'; snapshot.mkdir()
        # Archive the two consolidated files; no legacy support files are needed.
        source_paths = [Path(__file__)]
        launcher = Path(__file__).with_name('code.sh')
        if launcher.exists():
            source_paths.append(launcher)
        hashes = {}
        for path in source_paths:
            shutil.copy2(path, snapshot/path.name)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with mrcfile.mmap(args.input, mode='r') as source:
            if source.data.ndim != 3 or source.data.shape[1:] != (64, 64):
                raise ValueError('Expected the 64 x 64 individual CWF stack')
            total = len(source.data); n = total if args.limit is None else args.limit
            if not 1 <= n <= total: raise ValueError('Invalid particle limit')
            images = source.data[:n]; apix = float(source.voxel_size.x)
            masks = np.empty((n, 4, 64, 64), dtype=bool)
            vectors = np.lib.format.open_memmap(out/'vectors.npy', mode='w+', dtype=np.float32, shape=(n, 8*64*64))
            empty = np.zeros(n, dtype=bool)
            seed_yx = np.empty((n, 2), dtype=np.int16)
            summary = [Counter() for _ in SETTINGS['thresholds']]
            areas = np.zeros(4); overlaps = np.zeros(4); finite_overlaps = np.zeros(4, int)
            boundary = np.zeros(4, int)
            halo_changed = 0; halo_removed_pixels = 0
            log.info('Processing %d particles: four thresholds, outside factor %.2f, eight orientations', n, OUTSIDE_FACTOR)
            fields = ['source_index_0based', 'threshold', 'status', 'area_pixels', 'seed_score',
                      'touches_growth_boundary', 'iou_with_selected']
            with mrcfile.new_mmap(out/'particles_dimmed.mrc', shape=(n, 64, 64), mrc_mode=2) as dimmed_mrc, \
                    (out/'measurements.csv').open('w', newline='') as handle:
                dimmed_mrc.set_image_stack(); dimmed_mrc.voxel_size = apix
                dimmed_mrc.header.label[0] = b'Individual CWF; weakest central mask; outside pixels multiplied by 0.15'
                dimmed_mrc.header.nlabl = 1
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for row, image in enumerate(images):
                    result = segment_center_particle(image, **SETTINGS)
                    if args.ignore_halo:
                        assert result['masks'][0][result['core_mask']].all(), f'Lost bright core at {row}'
                        assert np.all(result['masks'] <= result['original_masks']), f'Mask expanded at {row}'
                        removed = int(result['halo_removed'][0])
                        halo_changed += removed > 0; halo_removed_pixels += removed
                    masks[row] = result['masks']; chosen = masks[row, 0]
                    empty[row] = not chosen.any(); seed_yx[row] = result['seed']
                    dimmed, views = oriented_particle_vectors(image, chosen, ANGLES, OUTSIDE_FACTOR)
                    dimmed_mrc.data[row] = dimmed; vectors[row] = views.ravel()
                    for j, (record, mask) in enumerate(zip(result['rows'], result['masks'])):
                        overlap = mask_iou(chosen, mask)
                        writer.writerow(dict(source_index_0based=row, **record, iou_with_selected=overlap))
                        summary[j][record['status']] += 1
                        areas[j] += record['area_pixels']; boundary[j] += record['touches_growth_boundary']
                        if np.isfinite(overlap): overlaps[j] += overlap; finite_overlaps[j] += 1
                    if (row+1) % 5000 == 0: log.info('Processed %d / %d', row+1, n)
                vectors.flush(); dimmed_mrc.update_header_stats()
            np.savez_compressed(out/'masks.npz', masks=masks, source_indices=np.arange(n),
                                thresholds=SETTINGS['thresholds'], selected_threshold_index=0)
            np.savez_compressed(out/'metadata.npz', source_indices=np.arange(n), angles_degrees=ANGLES,
                                image_shape=[64, 64], empty_masks=empty, seed_yx=seed_yx)
            settings = dict(source_mrc=str(args.input.resolve()), source_page=PAGE, image_count=n,
                            input_pixel_size_angstrom=apix, segmentation=SETTINGS,
                            selected_threshold_index=0, dimming_threshold_index=0, outside_factor=OUTSIDE_FACTOR,
                            angles_degrees=ANGLES.tolist(), vector_layout='angle, row, column (C order)',
                            interpolation='bilinear', rotation_background=0., reshape=False,
                            intensity_transform='original CWF inside mask; original CWF * 0.15 outside; no offset correction',
                            ignore_halo=args.ignore_halo,
                            halo_settings=dict(protected_core_threshold=1.1, protect_margin_pixels=2.,
                                               radial_quantile=.75, radial_smoothing=.8, neck_radius_pixels=1) if args.ignore_halo else None,
                            rotation_invariant=False)
            (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
            (out/'particles_cwf_original.mrc').symlink_to(args.input.resolve())
            review_refs = [16998, 998, *REFERENCES, 31911] if args.ignore_halo else REFERENCES
            previews = [i for i in review_refs if i < n] or list(range(min(5, n)))
            segmentation_plot(images, previews, out, '01_known_examples_segmentation.png')
            random_ids = np.sort(np.random.default_rng(1652).choice(n, min(6, n), replace=False))
            segmentation_plot(images, random_ids, out, '02_random_segmentation.png')
            orientation_plot(images, masks, previews[:3], out)
            # Verify disk outputs independently against source data for a fixed sample.
            checked = np.unique(np.r_[previews, np.linspace(0, n-1, min(64, n), dtype=int)])
            saved_vectors = np.load(out/'vectors.npy', mmap_mode='r')
            with mrcfile.mmap(out/'particles_dimmed.mrc', mode='r') as saved:
                for row in checked:
                    original = images[row]; mask = masks[row, 0]
                    actual = saved.data.reshape(-1, 64, 64)[row]
                    np.testing.assert_array_equal(actual[mask], original[mask])
                    np.testing.assert_array_equal(actual[~mask], original[~mask]*OUTSIDE_FACTOR)
                    views = saved_vectors[row].reshape(8, 64, 64)
                    np.testing.assert_array_equal(views[0], actual)
                    for j, turns in [(2, 1), (4, 2), (6, 3)]:
                        np.testing.assert_array_equal(views[j], np.rot90(actual, turns))
            for start in range(0, n, 512):
                if not np.isfinite(saved_vectors[start:start+512]).all(): raise ValueError('Non-finite vector values')
            with np.load(out/'masks.npz') as saved:
                np.testing.assert_array_equal(saved['masks'], masks)
                np.testing.assert_array_equal(saved['source_indices'], np.arange(n))
            thresholds = [dict(threshold=t, status_counts=dict(summary[j]), mean_area_pixels=float(areas[j]/n),
                               growth_boundary_count=int(boundary[j]),
                               mean_iou_with_selected=float(overlaps[j]/finite_overlaps[j]) if finite_overlaps[j] else None)
                          for j, t in enumerate(SETTINGS['thresholds'])]
            report = dict(particles=n, vector_shape=list(vectors.shape), vector_bytes=int(vectors.nbytes),
                          ignore_halo=args.ignore_halo, halo_changed_particles=int(halo_changed),
                          halo_removed_pixels=int(halo_removed_pixels), all_bright_cores_preserved=bool(args.ignore_halo),
                          empty_weakest_masks=int(empty.sum()), thresholds=thresholds, source_sha256=hashes,
                          known_examples={str(i): dict(empty_mask=bool(empty[i]), area_pixels=int(masks[i, 0].sum()),
                                                       seed_yx=seed_yx[i].tolist()) for i in previews},
                          preview_indices=random_ids.tolist(), validation=dict(checked_particles=len(checked),
                          unchanged_inside_mask=True, exact_outside_attenuation=True, cardinal_rotations=True,
                          finite_vectors=True, saved_masks_and_indices=True),
                          limitations=['Masks are threshold-based bright regions, not validated biological outlines.',
                                       'Connected noise may enter masks; faint disconnected arms may be attenuated.',
                                       'Eight concatenated views are not rotation invariant or aligned.',
                                       'Rotating on a fixed square grid clips corners and uses zero padding.',
                                       'Dimming original signed values moves them toward zero; it is not a new denoiser.',
                                       'No molecular class assignments or new PCA/clustering are made by this script.'])
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            (out/'README.txt').write_text(
                'NOTION POST-PROCESSING: sections 3.1–3.4\n\n'
                'Start: 01_known_examples_segmentation.png, 02_random_segmentation.png, 03_orientation_preview.png.\n'
                'Masks: masks.npz (particle, threshold, y, x), thresholds 0.1/0.3/0.5/1.1; selected=0.1.\n'
                'MRC: particles_dimmed.mrc, one complete image per original particle.\n'
                'Pixels inside the selected mask equal input CWF; outside multiply by 0.15.\n'
                'vectors.npy: disk-backed float32; load with np.load(path, mmap_mode="r").\n'
                'Reshape as (particles,8,64,64); angles 0,45,...315 degrees counterclockwise.\n'
                'metadata.npz: original zero-based indices, angles, empty-mask flags, seed positions.\n'
                'measurements.csv: one row per particle and threshold, including status and mask IoU.\n'
                'settings.json: all actual parameters and input provenance. report.json: counts and verification.\n'
                'Original input linked as particles_cwf_original.mrc. Source copies/hashes in source_code/ and report.json.\n'
                'Orientation concatenation is NOT rotation invariant. Masks can include noise or omit faint arms.\n'
                'This reproduces the requested page workflow; it does not claim accurate biological segmentation.\n'
                'Rerun: bash code.sh segment --input INPUT --output-dir NEW_DIR\n')
            if args.ignore_halo:
                with (out/'README.txt').open('a') as handle:
                    handle.write('\nHALO CORRECTION ENABLED: circular upper-quartile score profile, protected bright central lobes,\n'
                                 'and removal of narrow outer connections. This changes MASKS ONLY; retained CWF pixels are unchanged.\n'
                                 'Protect all sufficiently large central high-threshold components present in the permissive mask.\n'
                                 'Masks are subsets of the originals; all protected bright-core pixels checked for every particle.\n'
                                 'Faint thin features can still be lost. No biological segmentation accuracy is claimed.\n'
                                 'Halo-mode rerun: bash code.sh halo --input INPUT --output-dir NEW_DIR\n')
            log.info('COMPLETE: %d particles, %d empty weakest masks; vectors %.2f GiB', n, empty.sum(), vectors.nbytes/1024**3)
    return SimpleNamespace(segmentation_plot=segmentation_plot, orientation_plot=orientation_plot, main=main)


# ============================================================================
# Original module: test_particle_segmentation.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_test_particle_segmentation():
    module_description = """Scientific invariants for the Notion CWF segmentation protocol."""

    import unittest

    import numpy as np
    from scipy.ndimage import gaussian_filter

    ROTATION_ANGLES = _load_particle_segmentation().ROTATION_ANGLES
    SEGMENTATION_THRESHOLDS = _load_particle_segmentation().SEGMENTATION_THRESHOLDS
    mask_iou = _load_particle_segmentation().mask_iou
    oriented_particle_vectors = _load_particle_segmentation().oriented_particle_vectors
    segment_center_particle = _load_particle_segmentation().segment_center_particle
    standardize_particle_image = _load_particle_segmentation().standardize_particle_image


    class ParticleSegmentationTests(unittest.TestCase):
        def test_scores_are_standardized_on_the_smoothed_inscribed_circle(self):
            image = np.zeros((64, 64), dtype=np.float32)
            image[28:36, 28:36] = 5
            image[31, 32] = 20
            image[0, 0] = 1000  # Outside the reference disk.
            original = image.copy()
            result = segment_center_particle(image)
            np.testing.assert_array_equal(image, original)
            np.testing.assert_array_equal(result["smoothed"], gaussian_filter(image, sigma=1))
            yy, xx = np.indices(image.shape)
            reference = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 32 ** 2
            self.assertAlmostEqual(float(result["scores"][reference].mean()), 0, places=6)
            self.assertAlmostEqual(float(result["scores"][reference].std()), 1, places=6)
            self.assertAlmostEqual(result["reference_mean"], float(result["smoothed"][reference].mean(dtype=np.float64)))
            self.assertAlmostEqual(result["reference_std"], float(result["smoothed"][reference].std(dtype=np.float64)))
            self.assertEqual(result["masks"].shape, (4, 64, 64))
            self.assertEqual(tuple(row["threshold"] for row in result["rows"]), SEGMENTATION_THRESHOLDS)
            self.assertTrue(result["seed_region"][result["seed"]])
            dimmed, orientations = oriented_particle_vectors(image, result["masks"][1])
            np.testing.assert_array_equal(dimmed[result["masks"][1]], original[result["masks"][1]])
            np.testing.assert_array_equal(orientations[0], dimmed)

        def test_calibration_and_masks_are_invariant_to_offset_and_positive_gain(self):
            yy, xx = np.indices((64, 64))
            image = (10 * np.exp(-((yy - 30) ** 2 + (xx - 33) ** 2) / 20)).astype(np.float32)
            original = image.copy()
            shifted = image * 2.5 - 30
            first = segment_center_particle(image)
            second = segment_center_particle(shifted)
            np.testing.assert_array_equal(first["masks"], second["masks"])
            standardized = standardize_particle_image(image, first["reference_mean"], first["reference_std"])
            shifted_standardized = standardize_particle_image(shifted, second["reference_mean"], second["reference_std"])
            self.assertEqual(standardized.dtype, np.float32)
            np.testing.assert_allclose(standardized, shifted_standardized, atol=4e-6, rtol=2e-6)
            # Calibration uses unsmoothed pixels: a single positive linear transform.
            np.testing.assert_allclose(
                standardized,
                (image.astype(np.float64) - first["reference_mean"]) / first["reference_std"],
                rtol=1e-7,
            )
            dimmed, views = oriented_particle_vectors(standardized, first["masks"][0])
            shifted_dimmed, shifted_views = oriented_particle_vectors(shifted_standardized, second["masks"][0])
            np.testing.assert_allclose(dimmed, shifted_dimmed, atol=4e-6, rtol=2e-6)
            np.testing.assert_allclose(views, shifted_views, atol=4e-6, rtol=2e-6)
            np.testing.assert_array_equal(image, original)

        def test_negative_background_does_not_turn_bright_particle_black(self):
            yy, xx = np.indices((64, 64))
            image = (-20 + 10 * np.exp(-((yy - 31.5) ** 2 + (xx - 31.5) ** 2) / 20)).astype(np.float32)
            self.assertLess(float(image.max()), 0)
            result = segment_center_particle(image)
            mask = result["masks"][0]
            self.assertTrue(mask.any())
            calibrated = standardize_particle_image(image, result["reference_mean"], result["reference_std"])
            dimmed, orientations = oriented_particle_vectors(calibrated, mask)
            self.assertGreater(float(np.median(dimmed[mask])), 0)
            self.assertGreater(float(dimmed.max()), float(dimmed[~mask].max()))
            self.assertLess(float(np.median(dimmed[~mask])), 0)
            self.assertGreater(float(np.median(dimmed[mask])), float(np.median(dimmed[~mask])))
            np.testing.assert_array_equal(orientations[0], dimmed)

        def test_zero_reference_variation_creates_no_particle(self):
            image = np.full((64, 64), -7, dtype=np.float32)
            result = segment_center_particle(image)
            self.assertEqual(result["reference_std"], 0)
            standardized = standardize_particle_image(image, result["reference_mean"], result["reference_std"])
            self.assertEqual(standardized.dtype, np.float32)
            np.testing.assert_array_equal(standardized, 0)
            _, views = oriented_particle_vectors(standardized, result["masks"][0])
            np.testing.assert_array_equal(views, 0)

        def test_invalid_reference_statistics_are_rejected(self):
            image = np.zeros((8, 8), dtype=np.float32)
            for mean, std in [(np.nan, 1), (0, np.inf), (0, -1)]:
                with self.subTest(mean=mean, std=std), self.assertRaises(ValueError):
                    standardize_particle_image(image, mean, std)

        def test_weak_seed_and_tiny_component_return_empty_masks(self):
            constant = segment_center_particle(np.full((64, 64), 3, dtype=np.float32))
            self.assertFalse(constant["masks"].any())
            self.assertTrue(all(row["status"] == "weak_seed" for row in constant["rows"]))
            np.testing.assert_array_equal(constant["scores"], 0)
            tiny = np.zeros((64, 64), dtype=np.float32)
            tiny[31, 31] = 20
            result = segment_center_particle(tiny, sigma=0)
            self.assertFalse(result["masks"].any())
            self.assertTrue(all(row["status"] == "too_small" for row in result["rows"]))

        def test_seed_component_uses_diagonal_connectivity_and_excludes_other_particles(self):
            image = np.zeros((64, 64), dtype=np.float32)
            diagonal = np.arange(31, 39)
            image[diagonal, diagonal] = 8
            image[31, 31] = 20
            image[19:22, 40:43] = 8  # A disconnected component inside growth.
            result = segment_center_particle(image, sigma=0, thresholds=(0.1,))
            mask = result["masks"][0]
            self.assertEqual(result["seed"], (31, 31))
            self.assertEqual(np.count_nonzero(mask), 8)
            self.assertTrue(mask[diagonal, diagonal].all())
            self.assertFalse(mask[19:22, 40:43].any())

        def test_only_enclosed_small_holes_are_filled(self):
            image = np.zeros((64, 64), dtype=np.float32)
            image[24:40, 24:40] = 10
            image[29:31, 29:31] = 0  # Four pixels: fill.
            image[34:37, 34:37] = 0  # Nine pixels: retain.
            image[24:28, 26] = 0  # Connected to the exterior: retain.
            image[32, 32] = 20
            result = segment_center_particle(image, sigma=0, thresholds=(0.1,))
            mask = result["masks"][0]
            self.assertTrue(mask[29:31, 29:31].all())
            self.assertFalse(mask[34:37, 34:37].any())
            self.assertFalse(mask[24:28, 26].any())

        def test_growth_limit_and_boundary_diagnostic(self):
            image = np.zeros((64, 64), dtype=np.float32)
            image[30:34, 10:54] = 8
            image[31, 31] = 20
            result = segment_center_particle(image, sigma=0)
            self.assertTrue(all(row["touches_growth_boundary"] for row in result["rows"]))
            self.assertFalse(result["masks"][:, ~result["growth_region"]].any())
            yy, xx = np.indices(image.shape)
            expected = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= (64 * 0.35) ** 2
            np.testing.assert_array_equal(result["growth_region"], expected)

        def test_original_signs_and_intensities_are_preserved_inside_and_dimmed_outside(self):
            image = np.arange(-32, 32, dtype=np.float32).reshape(8, 8)
            mask = np.zeros_like(image, dtype=bool)
            mask[2:6, 2:6] = True
            original = image.copy()
            original_mask = mask.copy()
            dimmed, orientations = oriented_particle_vectors(image, mask)
            self.assertEqual(orientations.shape, (8, 8, 8))
            self.assertEqual(ROTATION_ANGLES, (0, 45, 90, 135, 180, 225, 270, 315))
            np.testing.assert_array_equal(dimmed[mask], original[mask])
            np.testing.assert_array_equal(dimmed[~mask], original[~mask] * 0.15)
            np.testing.assert_array_equal(np.sign(dimmed), np.sign(original))
            np.testing.assert_array_equal(orientations[0], dimmed)
            np.testing.assert_array_equal(orientations[2], np.rot90(dimmed))
            np.testing.assert_array_equal(orientations[4], np.rot90(dimmed, 2))
            np.testing.assert_array_equal(orientations[6], np.rot90(dimmed, 3))
            np.testing.assert_array_equal(image, original)
            np.testing.assert_array_equal(mask, original_mask)

        def test_empty_mask_still_dims_and_rotates_the_original(self):
            image = np.arange(81, dtype=np.float32).reshape(9, 9) - 40
            dimmed, orientations = oriented_particle_vectors(image, np.zeros_like(image, dtype=bool))
            np.testing.assert_array_equal(dimmed, image * 0.15)
            np.testing.assert_array_equal(orientations[2], np.rot90(dimmed))

        def test_iou_empty_masks_and_partial_overlap(self):
            first = np.array([[True, True], [False, False]])
            second = np.array([[True, False], [True, False]])
            self.assertAlmostEqual(mask_iou(first, second), 1 / 3)
            self.assertEqual(mask_iou(first, first), 1)
            empty = np.zeros((2, 2), dtype=bool)
            self.assertEqual(mask_iou(first, empty), 0)
            self.assertTrue(np.isnan(mask_iou(empty, empty)))

        def test_nonfinite_images_are_rejected(self):
            image = np.zeros((8, 8), dtype=np.float32)
            image[3, 3] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                segment_center_particle(image)
            with self.assertRaisesRegex(ValueError, "finite"):
                oriented_particle_vectors(image, np.ones_like(image, dtype=bool))
    return SimpleNamespace(ParticleSegmentationTests=ParticleSegmentationTests)


# ============================================================================
# Original module: test_halo_segmentation.py
# ============================================================================
@lru_cache(maxsize=None)
def _load_test_halo_segmentation():
    module_description = """Regression checks: remove 16998's outer arc without losing central lobes."""
    from pathlib import Path
    import sys
    import unittest
    import mrcfile
    import numpy as np
    from scipy.ndimage import label
    segment_center_particle = _load_halo_segmentation().segment_center_particle


    class HaloTests(unittest.TestCase):
        def test_reported_ring_and_all_central_bright_lobes(self):
            # Historical regression needs the original CWF stack, not arbitrary data.
            if REFERENCE_STACK is None:
                self.skipTest('Pass test --reference-stack to run the historical regression')
            source = REFERENCE_STACK
            with mrcfile.mmap(source) as data:
                for i in [16998, 998, 39304, 21949, 17325, 16637, 21528, 31911]:
                    image = data.data[i]; before = image.copy()
                    result = segment_center_particle(image)
                    old, new = result['original_masks'][0], result['masks'][0]
                    np.testing.assert_array_equal(image, before)
                    self.assertTrue(np.all(new <= old))
                    self.assertTrue(new[result['core_mask']].all())
                    if i == 16998:
                        _, cores = label(result['core_mask'], structure=np.ones((3, 3)))
                        self.assertGreaterEqual(cores, 2, 'Protect both separated bright lobes')
                        y, x = np.indices(image.shape)
                        # The visible lower ring lies at radii 18–22. The lower
                        # protein lobe also extends past radius 14, so auditing
                        # everything beyond 14 incorrectly counts its bright edge.
                        # This ROI is a regression audit, NOT an algorithm cutoff.
                        arc = (np.hypot(y-31.5, x-31.5) >= 18) & (y > 40)
                        old_arc, new_arc = int((old & arc).sum()), int((new & arc).sum())
                        print(f'16998: lower-arc mask pixels {old_arc} -> {new_arc}; all {int(result["core_mask"].sum())} bright-core pixels retained')
                        self.assertGreater(old_arc, 20)
                        self.assertEqual(new_arc, 0)

        def test_long_bright_arms_are_preserved_without_a_radial_cutoff(self):
            y, x = np.indices((64, 64)); y=y-31.5; x=x-31.5
            image = np.zeros((64, 64))
            for angle in np.deg2rad([45,135,225,315]):
                along=x*np.cos(angle)+y*np.sin(angle); across=-x*np.sin(angle)+y*np.cos(angle)
                d2=across**2+(along-np.clip(along, 0, 18))**2
                image=np.maximum(image, 10*np.exp(-d2/4))
            result=segment_center_particle(image.astype(np.float32))
            bright = (image > 7) & (np.hypot(x,y) > 14)
            self.assertTrue(bright.any())
            self.assertTrue(result['masks'][0][bright].all())

        def test_round_central_particle_and_constant_input(self):
            y,x=np.indices((64,64)); radius=np.hypot(y-31.5,x-31.5)
            image=(10*np.exp(-radius**2/32)).astype(np.float32)
            result=segment_center_particle(image)
            self.assertTrue(result['masks'][0][radius<5].all())
            self.assertFalse(segment_center_particle(np.ones((64,64), dtype=np.float32))['masks'].any())
    return SimpleNamespace(HaloTests=HaloTests)

# Original module: pilot_halo.py (input and output are now configurable).
def run_pilot(args):
    from pathlib import Path
    import sys, json
    import numpy as np
    import mrcfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    segment_center_particle = _load_halo_segmentation().segment_center_particle
    oriented_particle_vectors = _load_particle_segmentation().oriented_particle_vectors

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    # Use the same settings without requiring an unrelated historical run folder.
    settings = dict(source_mrc=str(args.input), segmentation=dict(sigma=1., seed_radius_fraction=.18,
        growth_radius_fraction=.35, thresholds=[.1, .3, .5, 1.1], seed_threshold=2., min_area=8, max_hole_area=8))
    ids = [16998, 998, 39304, 21949, 17325, 16637, 21528, 31911]
    rows = {}
    with mrcfile.mmap(settings['source_mrc']) as source:
        fig, axes = plt.subplots(len(ids), 5, figsize=(15, 3*len(ids)), squeeze=False)
        for row, index in enumerate(ids):
            image = source.data[index]
            result = segment_center_particle(image, **settings['segmentation'])
            old, new = result['original_masks'][0], result['masks'][0]
            assert np.all(new <= old)
            assert new[result['core_mask']].all()
            low, high = np.percentile(image, [1, 99.5])
            for col in [0, 1, 2]: axes[row, col].imshow(image, cmap='gray', vmin=low, vmax=high)
            axes[row, 1].contour(old, levels=[.5], colors=['orange'])
            axes[row, 2].contour(new, levels=[.5], colors=['cyan'])
            for col, mask in [(3, old), (4, new)]:
                dimmed, _ = oriented_particle_vectors(image, mask, [0], .15)
                axes[row, col].imshow(dimmed, cmap='gray', vmin=low, vmax=high)
            for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
            axes[row, 0].set_ylabel(str(index))
            rows[str(index)] = dict(old_area=int(old.sum()), new_area=int(new.sum()),
                                    strong_core_area=int(result['core_mask'].sum()),
                                    removed=int(old.sum()-new.sum()))
        for ax, title in zip(axes[0], ['Original CWF', 'Old mask', 'Halo-corrected mask', 'Old dimming', 'New dimming']): ax.set_title(title)
        fig.tight_layout(); fig.savefig(out/'halo_comparison.png', dpi=140); plt.close(fig)
    (out/'report.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))

def main():
    global REFERENCE_STACK
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("cwf", "halo", "segment", "class-shapes", "lowpass", "test", "pilot", "inspect", "requirements"))
    # Delegate each original CLI verbatim, including its --help and defaults.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    command = parser.parse_args(sys.argv[1:2]).command
    rest = sys.argv[2:]
    if command == "requirements":
        if rest:
            parser.error("requirements takes no arguments")
        print(RECORDED_REQUIREMENTS, end="")
        return
    if command == "test":
        tests_parser = argparse.ArgumentParser(description="Original synthetic tests and optional historical regression.")
        tests_parser.add_argument("--reference-stack", type=Path)
        args = tests_parser.parse_args(rest)
        REFERENCE_STACK = args.reference_stack
        suite = unittest.TestSuite()
        for cls in (_load_test_particle_segmentation().ParticleSegmentationTests,
                    _load_test_halo_segmentation().HaloTests):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if command in ("pilot", "inspect"):
        sub = argparse.ArgumentParser(description="Pilot diagnostics" if command == "pilot" else "Inspect an MRC stack")
        sub.add_argument("--input", type=Path, required=True)
        if command == "pilot":
            sub.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "halo_pilot")
        else:
            sub.add_argument("--index", type=int, default=16998)
        args = sub.parse_args(rest)
        if command == "pilot":
            run_pilot(args)
        else:
            # Original guide's MRC inspection example, with configurable path.
            import mrcfile
            with mrcfile.mmap(args.input, mode="r") as stack:
                print(stack.data.shape)  # Historical full output: (57107, 64, 64).
                particle = stack.data[args.index].copy()  # Original zero-based slice.
                print(f"Slice {args.index}: shape={particle.shape}, min={particle.min()}, max={particle.max()}")
        return
    loaders = {"cwf": _load_cwf_denoise, "halo": _load_notion_postprocess,
               "segment": _load_notion_postprocess, "class-shapes": _load_class_shapes,
               "lowpass": _load_denoise_particles}
    sys.argv = [f"{sys.argv[0]} {command}", *rest]
    if command == "halo" and "--ignore-halo" not in rest:
        # Original flag means ENABLE halo rejection despite its ambiguous name.
        sys.argv.append("--ignore-halo")
    loaders[command]().main()


if __name__ == "__main__":
    main()

