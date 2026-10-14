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
                lowpass_checkpoint=str(result.get("lowpass_checkpoint", "")),
                lowpass_sigma_pixels=float(result.get("lowpass_sigma_pixels", np.nan)))
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
