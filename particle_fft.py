"""ID-preserving Fourier descriptors of the Section 3 yellow growth circle."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def circle_fft(images, radius_fraction, taper_pixels=2.0, remove_mean=True):
    """Return shifted complex FFT and real log-magnitude features, without row scaling."""
    images = np.asarray(images)
    if (images.ndim != 3 or not len(images) or images.shape[1] != images.shape[2]
            or not np.isfinite(images).all()):
        raise ValueError("Expected a nonempty finite stack of square images")
    if not np.isfinite(radius_fraction) or not 0 < radius_fraction < .5:
        raise ValueError("radius_fraction must be in (0, 0.5)")
    size = images.shape[1]
    radius = size * radius_fraction
    if not np.isfinite(taper_pixels) or not 0 <= taper_pixels <= radius:
        raise ValueError("taper_pixels must be between zero and the circle radius")
    if not isinstance(remove_mean, (bool, np.bool_)):
        raise ValueError("remove_mean must be Boolean")
    yy, xx = np.indices((size, size))
    distance = np.hypot(xx-(size-1)/2, yy-(size-1)/2)
    circle = distance <= radius
    window = circle.astype(float)
    if taper_pixels:
        transition = np.clip((radius-distance)/taper_pixels, 0, 1)
        window *= .5 - .5*np.cos(np.pi*transition)
    if not window.any():
        raise ValueError("Circle/window contains no pixels")
    means = ((images*window).sum(axis=(1, 2))/window.sum()
             if remove_mean else np.zeros(len(images)))
    windowed = (images-means[:, None, None])*window
    spectra = np.fft.fftshift(np.fft.fft2(windowed, axes=(-2, -1), norm="ortho"),
                             axes=(-2, -1))
    log_magnitude = np.log1p(np.abs(spectra)).astype(np.float32)
    return dict(vectors=log_magnitude.reshape(len(images), -1),
                spectra=spectra.astype(np.complex64), images=log_magnitude,
                windowed_images=windowed.astype(np.float32), circle_mask=circle,
                window=window.astype(np.float32), removed_means=means,
                frequencies=np.fft.fftshift(np.fft.fftfreq(size)))


def save_circle_fft(root, segmentation_directory, taper_pixels=2.0, remove_mean=True):
    """Read the exact saved segmentation source/IDs; publish a validated FFT bundle."""
    root = Path(root).resolve()
    seg = Path(segmentation_directory).resolve()
    settings = json.loads((seg/"settings.json").read_text())
    with np.load(seg/"masks.npz", allow_pickle=False) as saved:
        names = saved["image_names"].astype(str)
        weakest = (int(saved["selected_threshold_index"]) if settings.get("method") in ("outside_in", "boundary_edges")
                   else int(np.argmin(saved["thresholds"])))
        empty = ~saved["masks"][:, weakest].any(axis=(1, 2))
    source = Path(settings["source_checkpoint"])
    with np.load(source/"cwf_result.npz", allow_pickle=False) as saved:
        source_names = saved["image_names"].astype(str).tolist()
        if len(set(source_names)) != len(source_names) or len(set(names)) != len(names):
            raise ValueError("Duplicate image IDs")
        lookup = {name: i for i, name in enumerate(source_names)}
        try:
            rows = [lookup[name] for name in names]
        except KeyError as error:
            raise ValueError(f"Segmentation ID missing in source: {error.args[0]}") from error
        images = saved["denoised"][rows]
    result = circle_fft(images, settings["growth_radius_fraction"], taper_pixels, remove_mean)
    folder = Path(tempfile.mkdtemp(prefix="particle_fft_circle_", dir=root))
    np.save(folder/"vectors.npy", result.pop("vectors"))
    # Log-magnitude images are a reshape of vectors.npy, not a duplicated array.
    result.pop("images")
    np.savez_compressed(folder/"fft_circle.npz", **result, image_names=names, empty_masks=empty)
    (folder/"settings.json").write_text(json.dumps(dict(
        schema_version=1, dataset="fft_circle", source_checkpoint=str(source),
        segmentation_directory=str(seg), source=settings["source"],
        radius_fraction=settings["growth_radius_fraction"], taper_pixels=float(taper_pixels),
        remove_weighted_mean=bool(remove_mean), fft_norm="ortho", fft_shifted=True,
        descriptor="log1p(abs(FFT2(window * (original CWF - weighted mean))))",
        feature_layout="frequency row, frequency column (C order)",
        frequency_units="cycles per source image pixel", image_shape=list(images.shape[1:]),
        n_features=int(np.prod(images.shape[1:])), row_normalization="none",
        support="yellow growth circle, independently of particle mask",
    ), indent=2)+"\n")
    loaded = load_fft_results(root, folder)
    pointer = root/"latest_fft_circle.json"
    temp = pointer.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(directory=folder.name))+"\n")
    os.replace(temp, pointer)
    return loaded


def load_fft_results(root, directory=None):
    root = Path(root).resolve()
    if directory is None:
        pointer = root/"latest_fft_circle.json"
        if not pointer.exists():
            raise FileNotFoundError("Run Section 3.9 to save fft_circle first")
        directory = json.loads(pointer.read_text())["directory"]
    folder = Path(directory)
    if not folder.is_absolute():
        folder = root/folder
    settings = json.loads((folder/"settings.json").read_text())
    with np.load(folder/"fft_circle.npz", allow_pickle=False) as saved:
        result = {k: saved[k].copy() for k in saved.files}
    vectors = np.load(folder/"vectors.npy", mmap_mode="r", allow_pickle=False)
    shape = tuple(settings["image_shape"])
    if (vectors.shape != (len(result["image_names"]), int(np.prod(shape)))
            or result["circle_mask"].shape != shape
            or result["empty_masks"].shape != (len(vectors),)
            or not np.isfinite(vectors).all()):
        raise ValueError("FFT feature dimensions or values are invalid")
    return dict(result, vectors=vectors, images=vectors.reshape(len(vectors), *shape),
                image_shape=shape, settings=settings, directory=folder)


def concatenate_by_image_id(base_vectors, base_names, fft_bundle, source_checkpoint):
    """Append FFT descriptors in base ID order; reject missing IDs or a different source."""
    base_vectors = np.asarray(base_vectors)
    names = np.asarray(base_names).astype(str).tolist()
    other_names = fft_bundle["image_names"].astype(str).tolist()
    if (base_vectors.ndim != 2 or len(base_vectors) != len(names)
            or not np.isfinite(base_vectors).all()):
        raise ValueError("Base features must be finite with one row per image ID")
    if len(set(names)) != len(names) or len(set(other_names)) != len(other_names):
        raise ValueError("Duplicate image IDs")
    if Path(source_checkpoint).resolve() != Path(fft_bundle["settings"]["source_checkpoint"]).resolve():
        raise ValueError("Feature blocks use different source checkpoints")
    lookup = {name: i for i, name in enumerate(other_names)}
    try:
        rows = [lookup[name] for name in names]
    except KeyError as error:
        raise ValueError(f"Missing FFT image ID: {error.args[0]}") from error
    return np.concatenate([base_vectors, fft_bundle["vectors"][rows]], axis=1)
