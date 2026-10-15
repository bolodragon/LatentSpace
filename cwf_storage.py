"""Pickle-free checkpoints for the CWF notebook; one directory per saved run."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np

KEYS = ('input', 'denoised', 'noise_input', 'noise_denoised')


def project_directory():
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / 'LatentSpace_data').is_dir():
            return parent
    raise FileNotFoundError('Cannot locate LatentSpace_data from the current directory')


def _atomic_npz(path, **values):
    temp = path.with_suffix('.tmp')
    try:
        with temp.open('wb') as handle:
            np.savez_compressed(handle, **values)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_cwf_result(result, root):
    arrays = {key: np.asarray(result[key]) for key in KEYS}
    if any(a.ndim != 3 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Expected finite 3D CWF stacks')
    if arrays['input'].shape != arrays['denoised'].shape or arrays['noise_input'].shape != arrays['noise_denoised'].shape:
        raise ValueError('Input/output dimensions disagree')
    if len(result['image_names']) != len(arrays['denoised']):
        raise ValueError('Particle names do not match image rows')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='cwf_saved_', dir=root))
    _atomic_npz(folder/'cwf_result.npz', **arrays,
                image_names=np.asarray(result['image_names'], dtype=str),
                pixel_size_angstrom=float(result['pixel_size_angstrom']),
                image_shape=np.asarray(arrays['denoised'].shape[1:]),
                run_id=folder.name, schema_version=1,
                particle_source=str(result.get("particle_source", "local_cwf")),
                input_is_processed_reference=bool(result.get("input_is_processed_reference", False)),
                noise_control_source=str(result.get("noise_control_source", "local_cwf")),
                lowpass_checkpoint=str(result.get("lowpass_checkpoint", "")),
                lowpass_sigma_pixels=float(result.get("lowpass_sigma_pixels", np.nan)))
    if "normalization" in result:
        (folder / "normalization.json").write_text(json.dumps(result["normalization"], indent=2) + "\n")
        _atomic_npz(folder / "normalization_factors.npz",
                    offsets=result["normalization_offsets"], scales=result["normalization_scales"])
    # Attach provenance to this exact in-memory result for the subsequent crop save.
    result['_checkpoint_directory'] = str(folder)
    return folder


def save_cwf_circles(result, pixels, mask, diameter):
    if '_checkpoint_directory' not in result:
        raise RuntimeError('Run the save cell in 2.2 for this cwf_result before saving circles')
    folder = Path(result['_checkpoint_directory'])
    mask = np.asarray(mask, dtype=bool)
    # Check against current full outputs, preventing stale crop arrays being saved.
    if mask.shape != result['denoised'].shape[1:] or not mask.any():
        raise ValueError('Circle mask does not match the CWF images')
    for key in KEYS:
        if not np.array_equal(np.asarray(pixels[key]), np.asarray(result[key])[:,mask]):
            raise ValueError(f'{key}: cropped pixels are stale; rerun 2.4 extraction')
    with np.load(folder/'cwf_result.npz', allow_pickle=False) as full:
        if not np.array_equal(full['image_names'], np.asarray(result['image_names'], dtype=str)):
            raise ValueError('Saved and current particle names disagree')
    _atomic_npz(folder/'cwf_circles.npz', **pixels, mask=mask, diameter=float(diameter),
                image_names=np.asarray(result['image_names'], dtype=str),
                pixel_size_angstrom=float(result['pixel_size_angstrom']),
                run_id=folder.name, schema_version=1)
    # Advertise only runs for which BOTH files are complete.
    pointer = folder.parent/'latest_cwf_saved.json'
    temp = pointer.with_suffix('.tmp')
    temp.write_text(json.dumps({'directory':folder.name}, indent=2)+'\n')
    os.replace(temp, pointer)
    return folder/'cwf_circles.npz'


def select_cwf_directory(root, source, directory=None):
    """Select a complete checkpoint of the requested notebook input source."""
    if source not in ("own", "camille"):
        raise ValueError('cwf_input_source must be "own" or "camille".')
    root = Path(root)

    def matches(folder):
        if not all((folder / name).is_file() for name in ("cwf_result.npz", "cwf_circles.npz")):
            return False
        with np.load(folder / "cwf_result.npz", allow_pickle=False) as saved:
            external = bool(saved["input_is_processed_reference"]) if "input_is_processed_reference" in saved else False
        return external == (source == "camille")

    if directory is not None:
        folder = Path(directory)
        if not folder.is_absolute():
            folder = root / folder
        if not matches(folder):
            raise ValueError(f"{folder} is not a complete {source!r} checkpoint. Check cwf_input_source and cwf_saved_directory.")
        return folder
    candidates = [folder for folder in root.glob("cwf_saved_*") if matches(folder)]
    if not candidates:
        preparation = "2.2 and 2.3" if source == "own" else "2.5"
        raise FileNotFoundError(f"No complete {source!r} checkpoint found. Run section(s) {preparation} first.")
    return max(candidates, key=lambda folder: (folder / "cwf_circles.npz").stat().st_mtime_ns)


def load_cwf_circles(root, directory=None):
    root = Path(root)
    if directory is None:
        pointer = root/'latest_cwf_saved.json'
        if not pointer.exists():
            raise FileNotFoundError('No complete checkpoint. Run the save cells in 2.2 and 2.4 once.')
        folder = root/json.loads(pointer.read_text())['directory']
    else:
        folder = Path(directory)
        if not folder.is_absolute(): folder = root/folder
    # NPZ access is lazy: read metadata from full results, not the large full stacks.
    with np.load(folder/'cwf_result.npz', allow_pickle=False) as full, np.load(folder/'cwf_circles.npz', allow_pickle=False) as crop:
        if str(full['run_id']) != str(crop['run_id']):
            raise ValueError('Full and cropped checkpoints belong to different runs')
        names = full['image_names'].tolist()
        if names != crop['image_names'].tolist():
            raise ValueError('Saved image order differs between checkpoints')
        mask = crop['mask'].astype(bool)
        if tuple(full['image_shape']) != mask.shape:
            raise ValueError('Saved mask and full image dimensions disagree')
        pixels = {key:crop[key].copy() for key in KEYS}
        if any(a.ndim != 2 or a.shape[1] != mask.sum() or not np.isfinite(a).all() for a in pixels.values()):
            raise ValueError('Invalid circle-pixel arrays')
        if len(names) != len(pixels['denoised']):
            raise ValueError('Saved names and cropped particle rows disagree')
        if pixels['input'].shape != pixels['denoised'].shape or pixels['noise_input'].shape != pixels['noise_denoised'].shape:
            raise ValueError('Saved input/output dimensions disagree')
        diameter = float(crop['diameter'])
    return pixels, mask, names, diameter, folder


def import_external_cwf(path, previous_result, count=4000, seed=0):
    """Import a reproducible random subset; retain local noise controls.

    External row order has no verified raw/STAR mapping. The input slot therefore
    holds a processed reference copy, NOT a matched raw image.
    """
    import mrcfile
    from aspire.image import Image
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    with mrcfile.mmap(str(path), mode="r") as mrc:
        if mrc.data.ndim != 3 or mrc.data.shape[1] != mrc.data.shape[2] or len(mrc.data) < count:
            raise ValueError("Need a square MRC stack with at least count images")
        indices = np.sort(np.random.default_rng(seed).choice(len(mrc.data), size=count, replace=False))
        images = np.array(mrc.data[indices], dtype=np.float32, copy=True)
        spacing = np.array([float(mrc.voxel_size.x), float(mrc.voxel_size.y)])
    if not np.isfinite(images).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0) or not np.allclose(spacing, spacing[0]):
        raise ValueError("Invalid external pixels or MRC pixel size")
    size = images.shape[-1]
    result = dict(input=images.copy(), denoised=images,
                  image_names=[f"{i+1:06d}@{Path(path).resolve()}" for i in indices],
                  pixel_size_angstrom=float(spacing[0]), particle_source=str(Path(path).resolve()),
                  input_is_processed_reference=True, noise_control_source="local_cwf_separate_processing")
    for key in ("noise_input", "noise_denoised"):
        a = np.asarray(previous_result[key], dtype=np.float32)
        if a.ndim != 3 or a.shape[1] != a.shape[2] or not len(a) or not np.isfinite(a).all():
            raise ValueError("Need existing finite local noise controls")
        if a.shape[-1] < size or not np.isclose(a.shape[-1]*float(previous_result["pixel_size_angstrom"]), size*spacing[0], rtol=1e-3):
            raise ValueError("Noise controls must cover the same physical box and have >= external resolution")
        result[key] = a.copy() if a.shape[-1] == size else Image(a).downsample(size).asnumpy().copy()
    return result


def normalize_camille_images(images, diameter=None):
    """Standardize each image using only its centered circle's mean and std.

    Returned offsets are in input units: output = (images - offsets) / scales.
    """
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3 or images.shape[1] != images.shape[2] or not len(images) or not np.isfinite(images).all():
        raise ValueError("Expected a nonempty finite stack of square images")
    size = images.shape[-1]
    diameter = float(size if diameter is None else diameter)
    if not np.isfinite(diameter) or not 0 < diameter <= size:
        raise ValueError("Circle diameter must be positive and at most the image width")
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - (size - 1) / 2)**2 + (yy - (size - 1) / 2)**2 <= (diameter / 2)**2
    if mask.sum() < 2:
        raise ValueError("The center circle must contain at least two pixels")
    values = images[:, mask]
    offsets = values.mean(axis=1)
    scales = values.std(axis=1)
    tolerance = np.finfo(np.float64).eps * np.maximum(np.max(np.abs(values), axis=1), 1)
    invalid = ~np.isfinite(scales) | (scales <= tolerance)
    if invalid.any():
        raise ValueError(f"Negligible center variation for image rows {(np.flatnonzero(invalid) + 1).tolist()[:10]}")
    normalized = ((images - offsets[:, None, None]) / scales[:, None, None]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced non-finite pixels")
    return normalized, mask, offsets, scales


def load_cluster_cwf_images(directory, image_table):
    """Retrieve full, unrotated CWF outputs in exact cluster-table row order."""
    with np.load(Path(directory) / "cwf_result.npz", allow_pickle=False) as saved:
        names = saved['image_names'].astype(str).tolist()
        if len(set(names)) != len(names):
            raise ValueError('Source checkpoint contains duplicate particle image names')
        lookup = {name: row for row, name in enumerate(names)}
        sources = image_table['source'].to_numpy()
        if not np.isin(sources, ['particle', 'noise']).all():
            raise ValueError('Unknown image source in cluster table')
        particles = saved['denoised']
        images = np.empty((len(image_table), *particles.shape[1:]), dtype=particles.dtype)
        particle_rows = np.flatnonzero(sources == 'particle')
        try:
            indices = [lookup[str(name)] for name in image_table.iloc[particle_rows]['image_name']]
        except KeyError as error:
            raise ValueError(f'Particle {error.args[0]} is missing from the source checkpoint') from error
        images[particle_rows] = particles[indices]
        noise_rows = np.flatnonzero(sources == 'noise')
        if len(noise_rows):
            noise = saved['noise_denoised']
            raw_indices = image_table.iloc[noise_rows]['source_row_1based'].to_numpy()
            indices = raw_indices.astype(int) - 1
            if not np.array_equal(raw_indices, indices+1) or np.any(indices < 0) or np.any(indices >= len(noise)):
                raise ValueError('Noise indices do not match the source checkpoint')
            images[noise_rows] = noise[indices]
    if not np.isfinite(images).all():
        raise ValueError('Source CWF images contain non-finite pixels')
    return images
