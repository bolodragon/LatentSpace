"""Exploratory center-seeded segmentation; does not alter CWF/PCA inputs."""
import numpy as np
from scipy import ndimage as ndi


def segment_center_particle(image, sigma=1.0, seed_radius_fraction=0.18,
                            growth_radius_fraction=0.35, thresholds=(1.0, 1.5, 2.0),
                            seed_threshold=2.5, min_area=8, max_hole_area=8,
                            threshold_mode="standardized"):
    """Grow from a central seed using standardized or seed-relative contrast."""
    image = np.asarray(image, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or not np.isfinite(image).all():
        raise ValueError("Expected a finite square image")
    if (not np.isfinite(sigma) or sigma < 0 or
            not 0 < seed_radius_fraction <= growth_radius_fraction < .5):
        raise ValueError("Need sigma >= 0 and 0 < seed radius <= growth radius < 0.5")
    if (thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all()
            or np.any(thresholds <= 0) or np.any(np.diff(thresholds) <= 0)
            or not np.isfinite(seed_threshold) or seed_threshold <= 0):
        raise ValueError("Use increasing positive thresholds and a positive seed threshold")
    if threshold_mode not in ("relative", "standardized"):
        raise ValueError("threshold_mode must be relative or standardized")
    if threshold_mode == "relative" and np.any(thresholds > 1):
        raise ValueError("Relative thresholds must be fractions in (0, 1]")
    if threshold_mode == "standardized" and thresholds[-1] > seed_threshold:
        raise ValueError("Standardized thresholds cannot exceed the seed threshold")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (min_area, max_hole_area)) or min_area < 1 or max_hole_area < 0:
        raise ValueError("Use integer min_area >= 1 and max_hole_area >= 0")
    size = image.shape[0]
    yy, xx = np.ogrid[:size, :size]
    radius2 = (xx-(size-1)/2)**2 + (yy-(size-1)/2)**2
    seed_region = radius2 <= (size*seed_radius_fraction)**2
    growth_region = radius2 <= (size*growth_radius_fraction)**2
    support = radius2 <= (size/2)**2
    smoothed = ndi.gaussian_filter(image, sigma=sigma)
    background = float(smoothed[support].mean())
    scale = smoothed[support].std()
    scores = np.zeros_like(smoothed)
    if scale > np.finfo(float).eps * max(1, np.abs(smoothed[support]).max()):
        scores = (smoothed-background) / scale
    seed = np.unravel_index(np.argmax(np.where(seed_region, scores, -np.inf)), image.shape)
    seed_ok = scores[seed] >= seed_threshold
    masks, rows = [], []
    connectivity = np.ones((3, 3), dtype=bool)
    boundary = growth_region & ~ndi.binary_erosion(growth_region)
    for threshold in thresholds:
        cutoff = float(threshold * scores[seed] if threshold_mode == "relative" else threshold)
        mask = np.zeros_like(image, dtype=bool)
        if seed_ok:
            labels, _ = ndi.label((scores >= cutoff) & growth_region, structure=connectivity)
            label = labels[seed]
            if label:
                mask = labels == label  # Discard disconnected bright regions.
            if mask.sum() < min_area:
                mask[:] = False
            elif max_hole_area:
                holes = ndi.binary_fill_holes(mask) & ~mask & growth_region
                hole_labels, _ = ndi.label(holes)
                sizes = np.bincount(hole_labels.ravel())
                fill = sizes <= max_hole_area
                fill[0] = False
                mask |= fill[hole_labels]
        masks.append(mask)
        rows.append(dict(threshold=float(threshold), threshold_mode=threshold_mode,
                         cutoff_standardized=cutoff, cutoff_intensity=float(background + cutoff*scale),
                         background_mean=background, seed_intensity=float(smoothed[seed]),
                         area_pixels=int(mask.sum()),
                         seed_score=float(scores[seed]),
                         touches_growth_boundary=bool(np.any(mask & boundary)),
                         status="ok" if mask.any() else ("weak_seed" if not seed_ok else "too_small")))
    return dict(smoothed=smoothed, scores=scores, masks=np.asarray(masks),
                seed=seed, seed_region=seed_region, growth_region=growth_region, rows=rows)


def mask_iou(first, second):
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else float('nan')


def oriented_particle_vectors(image, mask, angles, outside_factor=0.15):
    """Dim toward zero outside a mask; flatten rotations in angle-major order.

    Positive angles rotate counterclockwise on the displayed image. Rotation
    uses bilinear interpolation on the original grid, with zero outside it.
    """
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    angles = np.asarray(angles, dtype=float)
    if image.ndim != 2 or mask.shape != image.shape or not np.isfinite(image).all():
        raise ValueError("Need a finite image and matching mask")
    if not np.isfinite(outside_factor) or not 0 <= outside_factor <= 1:
        raise ValueError("outside_factor must be between 0 and 1")
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Need a nonempty list of finite rotation angles")
    dimmed = image * np.where(mask, 1.0, outside_factor).astype(np.float32)
    rotations = np.stack([
        dimmed.copy() if angle % 360 == 0 else
        ndi.rotate(dimmed, float(angle), reshape=False, order=1,
                   mode="constant", cval=0.0, prefilter=False)
        for angle in angles
    ])
    return dimmed, rotations.reshape(len(angles), -1)


def load_section3_results(root, directory=None):
    """Load a finalized Section 3 bundle; vectors remain memory-mapped."""
    import json
    from pathlib import Path
    root = Path(root).resolve()
    if directory is None:
        pointer = root / "latest_section3.json"
        if not pointer.exists():
            raise FileNotFoundError("No finalized Section 3 results. Run 3.6 first.")
        directory = json.loads(pointer.read_text())["directory"]
    folder = Path(directory)
    if not folder.is_absolute():
        folder = root / folder
    manifest = json.loads((folder / "section3.json").read_text())
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported Section 3 checkpoint version")
    vectors = np.load(folder / "vectors.npy", mmap_mode="r", allow_pickle=False)
    with np.load(folder / "metadata.npz", allow_pickle=False) as meta:
        names, angles = meta["image_names"].copy(), meta["angles_degrees"].copy()
        shape, empty = tuple(meta["image_shape"].tolist()), meta["empty_masks"].copy()
    with np.load(folder / "masks.npz", allow_pickle=False) as saved:
        masks, thresholds = saved["masks"].copy(), saved["thresholds"].copy()
        if not np.array_equal(names, saved["image_names"]):
            raise ValueError("Segmentation and orientation image order disagree")
    if (len(shape) != 2 or vectors.shape != (len(names), len(angles)*int(np.prod(shape)))
            or masks.shape != (len(names), len(thresholds), *shape)
            or empty.shape != (len(names),)):
        raise ValueError("Saved Section 3 dimensions disagree")
    orientation_images = vectors.reshape(len(names), len(angles), *shape)
    averaged_images = None
    if manifest["schema_version"] == 2:
        orientation_images = np.load(folder / "orientation_images.npy", mmap_mode="r", allow_pickle=False)
        averaged_images = np.load(folder / "averaged_images.npy", mmap_mode="r", allow_pickle=False)
        if (orientation_images.shape != (len(names), len(angles), *shape)
                or averaged_images.shape != (len(names), *shape)):
            raise ValueError("Saved image datasets have incorrect dimensions")
    return dict(directory=folder, vectors=vectors,
                masked_images=(np.load(folder / "masked_images.npy", mmap_mode="r", allow_pickle=False)
                               if (folder / "masked_images.npy").exists() else None),
                orientation_images=orientation_images, averaged_images=averaged_images,
                orientation_views=vectors.reshape(len(names), len(angles), -1),
                image_names=names, angles_degrees=angles, image_shape=shape,
                masks=masks, thresholds=thresholds, empty_masks=empty,
                settings=json.loads((folder / "settings.json").read_text()),
                segmentation_settings=json.loads((folder / "segmentation_settings.json").read_text()),
                measurements_path=folder / "measurements.csv", manifest=manifest)


def finalize_section3_results(directory):
    """Bundle segmentation with saved vectors and publish only after validation."""
    import json
    import os
    import shutil
    from pathlib import Path
    folder = Path(directory).resolve()
    settings = json.loads((folder / "settings.json").read_text())
    source = Path(settings["segmentation_directory"])
    for src, dest in (("masks.npz", "masks.npz"), ("measurements.csv", "measurements.csv"),
                      ("settings.json", "segmentation_settings.json")):
        shutil.copy2(source / src, folder / dest)
    if (source / "segmentation_preview.png").exists():
        shutil.copy2(source / "segmentation_preview.png", folder / "segmentation_preview.png")
    # Save explicit image datasets in batches, retaining vectors for Section 4.
    vectors = np.load(folder / "vectors.npy", mmap_mode="r", allow_pickle=False)
    with np.load(folder / "metadata.npz", allow_pickle=False) as meta:
        count = len(meta["image_names"])
        angles = meta["angles_degrees"]
        shape = tuple(meta["image_shape"].tolist())
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Saved rotation angles must be a nonempty finite vector")
    rotation_count = len(angles)
    if len(shape) != 2 or vectors.shape != (count, rotation_count * int(np.prod(shape))):
        raise ValueError("Vector dimensions do not match the saved orientations")
    image_view = vectors.reshape(count, rotation_count, *shape)
    orientation_temp = folder / "orientation_images.tmp.npy"
    average_temp = folder / "averaged_images.tmp.npy"
    rotated = np.lib.format.open_memmap(orientation_temp, mode="w+", dtype=np.float32,
                                      shape=image_view.shape)
    averaged = np.lib.format.open_memmap(average_temp, mode="w+", dtype=np.float32,
                                       shape=(count, *shape))
    for start in range(0, count, 32):
        block = image_view[start:start+32]
        if not np.isfinite(block).all():
            raise ValueError("Orientation images contain non-finite pixels")
        rotated[start:start+32] = block
        averaged[start:start+32] = block.mean(axis=1, dtype=np.float64)
    rotated.flush()
    averaged.flush()
    del rotated, averaged
    os.replace(orientation_temp, folder / "orientation_images.npy")
    os.replace(average_temp, folder / "averaged_images.npy")
    (folder / "section3.json").write_text(json.dumps(dict(
        schema_version=2, source_checkpoint=settings["source_checkpoint"],
        datasets={"orientation_images.npy": [count, rotation_count, *shape],
                  "averaged_images.npy": [count, *shape]},
        averaging="pixelwise arithmetic mean over saved rotations (axis 1)",
        description="Section 3 masks, measurements, and masked orientation vectors",
    ), indent=2) + "\n")
    result = load_section3_results(folder.parent, folder)
    pointer = folder.parent / "latest_section3.json"
    temp = pointer.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(directory=folder.name), indent=2) + "\n")
    os.replace(temp, pointer)
    return result


def rotation_angles(start_degrees=0.0, stop_degrees=360.0, count=8):
    """Evenly sample [start, stop), avoiding duplicate full-turn endpoints."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Rotation count must be a positive integer")
    if (not np.isfinite(start_degrees) or not np.isfinite(stop_degrees)
            or not 0 < stop_degrees - start_degrees <= 360):
        raise ValueError("Rotation range must have a positive span of at most 360 degrees")
    return np.linspace(start_degrees, stop_degrees, num=count, endpoint=False, dtype=float)


def mask_with_margin(image, mask, margin_pixels=2.0):
    """Keep pixels within Euclidean margin of a mask; zero everything else."""
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 2 or mask.shape != image.shape or not np.isfinite(image).all():
        raise ValueError("Need a finite image and matching 2D mask")
    if not np.isfinite(margin_pixels) or margin_pixels < 0:
        raise ValueError("Margin must be a finite nonnegative pixel distance")
    expanded = mask.copy()
    if mask.any() and margin_pixels > 0:
        expanded |= ndi.distance_transform_edt(~mask) <= margin_pixels
    return np.where(expanded, image, 0).astype(np.float32), expanded
