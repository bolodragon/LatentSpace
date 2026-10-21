"""Outside-in background removal, independent of central brightness."""
import numpy as np
from scipy import ndimage as ndi


def _disk(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x*x+y*y <= radius*radius


def remove_boundary_edges(edges, domain, boundary_band_pixels=2, connectivity=8,
                          edge_expansion_pixels=1, fill_removed_loops=True):
    """Remove boundary-contact edge components and their genuinely enclosed interiors.

    Label before expanding edges so display thickness cannot join unrelated
    components. The circle is never used as an edge to close an open contour.
    Everything else inside the domain is kept, regardless of brightness.
    """
    structure = ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    band = domain & (ndi.distance_transform_edt(domain) <= boundary_band_pixels)
    edges = np.asarray(edges, dtype=bool) & domain
    labels, count = ndi.label(edges, structure=structure)
    bad_ids = np.unique(labels[edges & band])
    bad_ids = bad_ids[bad_ids > 0]
    rejected = np.zeros_like(domain)
    enclosed = np.zeros_like(domain)
    hole_structure = ndi.generate_binary_structure(2, 1 if connectivity == 8 else 2)
    for label in bad_ids:
        component = labels == label
        if edge_expansion_pixels:
            component = ndi.binary_dilation(component, structure=_disk(edge_expansion_pixels)) & domain
        rejected |= component
        if fill_removed_loops:
            enclosed |= ndi.binary_fill_holes(component, structure=hole_structure) & ~component & domain
    removed = rejected | enclosed
    kept_edges = edges & ~removed
    mask = domain & ~removed
    cy, cx = (np.asarray(domain.shape)-1)//2
    return dict(mask=mask, boundary_band=band, contact_edges=edges & band,
                rejected_edges=rejected, removed_enclosures=enclosed,
                removed=removed, kept_edges=kept_edges,
                rejected_component_count=len(bad_ids), kept_component_count=count-len(bad_ids),
                center_removed=bool(removed[cy, cx]),
                status="empty_after_edge_removal" if not mask.any() else
                       "review_center_removed" if removed[cy, cx] else "ok")


def segment_boundary_edges(image, intensity_percentile=50, edge_percentiles=(60, 70, 80),
                           sigma=1.0, growth_radius_fraction=.35, boundary_band_pixels=2,
                           gap_closing_radius_pixels=1, edge_expansion_pixels=1,
                           connectivity=8, ridge_only=True, fill_removed_loops=True):
    """Public interface for the boundary-edge cleanup workflow in notebook 3.1–3.3."""
    if not isinstance(fill_removed_loops, (bool, np.bool_)):
        raise ValueError("fill_removed_loops must be Boolean")
    return segment_exterior(image, intensity_percentile=intensity_percentile,
        edge_percentiles=edge_percentiles, sigma=sigma, growth_radius_fraction=growth_radius_fraction,
        central_radius_fraction=min(.18, growth_radius_fraction),
        exterior_band_pixels=boundary_band_pixels,
        gap_closing_radius_pixels=gap_closing_radius_pixels,
        barrier_dilation_pixels=edge_expansion_pixels, connectivity=connectivity,
        ridge_only=ridge_only, removal_mode="boundary_edges", fill_removed_loops=fill_removed_loops)


def exterior_partition(barrier, domain, central_region, exterior_band_pixels=2,
                       connectivity=8, min_area=8, max_area_fraction=.8,
                       require_central_overlap=True, boundary_margin_pixels=1.0):
    """Flood from the domain boundary, then select enclosed components geometrically."""
    structure = ndi.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    # This band is inside the analysis circle, avoiding the artificial CWF support edge.
    band = domain & (ndi.distance_transform_edt(domain) <= exterior_band_pixels)
    free = domain & ~barrier
    seeds = band & free
    exterior = ndi.binary_propagation(seeds, structure=structure, mask=free)
    enclosed = free & ~exterior
    labels, count = ndi.label(enclosed, structure=structure)
    retained = np.zeros_like(domain)
    kept = 0
    if seeds.any():
        for label in range(1, count+1):
            component = labels == label
            area = int(component.sum())
            if (min_area <= area <= max_area_fraction*domain.sum()
                    and (not require_central_overlap or (component & central_region).any())):
                retained |= component
                kept += 1
    # Reattach only a small nearby edge rim, never an entire connected edge network.
    mask = retained.copy()
    if retained.any() and boundary_margin_pixels:
        mask |= (barrier & domain
                 & (ndi.distance_transform_edt(~retained) <= boundary_margin_pixels))
    return dict(mask=mask, exterior=exterior, enclosed=enclosed, retained=retained,
                exterior_seeds=seeds, exterior_band=band, component_count=kept,
                status=("no_exterior_seeds" if not seeds.any()
                        else "ok" if mask.any() else "no_retained_region"))


def segment_exterior(image, intensity_percentile=50, edge_percentiles=(60, 70, 80),
                     sigma=1.0, growth_radius_fraction=.35, central_radius_fraction=.18,
                     exterior_band_pixels=2, gap_closing_radius_pixels=1,
                     barrier_dilation_pixels=0, connectivity=8, min_area=8,
                     max_area_fraction=.8, require_central_overlap=True,
                     boundary_margin_pixels=1.0, ridge_only=True,
                     removal_mode="outside_in", fill_removed_loops=True):
    """Threshold whole-image percentiles; discard both dark and bright exterior regions.

    Intensity classifies background for inspection; it never gates the flood or
    selects foreground. Edge percentiles change the barrier and final masks.
    """
    image = np.asarray(image, dtype=float)
    if removal_mode not in ("outside_in", "boundary_edges"):
        raise ValueError("Unknown removal mode")
    percentiles = np.asarray(edge_percentiles, dtype=float)
    if (image.ndim != 2 or min(image.shape) < 3 or image.shape[0] != image.shape[1]
            or not np.isfinite(image).all()):
        raise ValueError("Expected a finite square image of width >= 3")
    if not np.isfinite(intensity_percentile) or not 0 <= intensity_percentile <= 100:
        raise ValueError("Intensity percentile must be in [0, 100]")
    if (percentiles.ndim != 1 or not len(percentiles) or not np.isfinite(percentiles).all()
            or np.any((percentiles < 0) | (percentiles > 100)) or np.any(np.diff(percentiles) <= 0)):
        raise ValueError("Edge percentiles must be strictly increasing and in [0, 100]")
    if (not np.isfinite(sigma) or sigma < 0
            or not 0 < central_radius_fraction <= growth_radius_fraction < .5):
        raise ValueError("Use sigma >= 0 and 0 < central radius <= growth radius < .5")
    for value, minimum in ((gap_closing_radius_pixels, 0), (barrier_dilation_pixels, 0), (min_area, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError("Morphology radii must be nonnegative integers; min_area a positive integer")
    if (not np.isfinite(exterior_band_pixels) or not 1 <= exterior_band_pixels < image.shape[0]*growth_radius_fraction
            or not np.isfinite(boundary_margin_pixels) or boundary_margin_pixels < 0
            or not np.isfinite(max_area_fraction) or not 0 < max_area_fraction <= 1):
        raise ValueError("Invalid exterior band, boundary margin, or maximum area fraction")
    if isinstance(connectivity, (bool, np.bool_)) or connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if not isinstance(require_central_overlap, (bool, np.bool_)):
        raise ValueError("require_central_overlap must be Boolean")
    if not isinstance(ridge_only, (bool, np.bool_)):
        raise ValueError("ridge_only must be Boolean")
    smoothed = ndi.gaussian_filter(image, sigma)
    if sigma:
        gy = ndi.gaussian_filter(image, sigma, order=(1, 0))
        gx = ndi.gaussian_filter(image, sigma, order=(0, 1))
    else:
        gy, gx = np.gradient(image)
    gradient = np.hypot(gx, gy)  # No seed or central-brightness normalization.
    # Thin along the gradient normal, not by altering the derivative image.
    yy, xx = np.indices(image.shape)
    ny = np.divide(gy, gradient, out=np.zeros_like(gy), where=gradient > 0)
    nx = np.divide(gx, gradient, out=np.zeros_like(gx), where=gradient > 0)
    ahead = ndi.map_coordinates(gradient, [yy+ny, xx+nx], order=1, mode="nearest")
    behind = ndi.map_coordinates(gradient, [yy-ny, xx-nx], order=1, mode="nearest")
    ridges = (gradient >= ahead) & (gradient >= behind)
    intensity_cutoff = float(np.percentile(image, intensity_percentile))
    dark = image <= intensity_cutoff
    size = image.shape[0]
    yy, xx = np.indices(image.shape)
    distance = np.hypot(xx-(size-1)/2, yy-(size-1)/2)
    domain = distance <= size*growth_radius_fraction
    central = distance <= size*central_radius_fraction
    boundary = domain & ~ndi.binary_erosion(domain)
    rows, details = [], []
    for percentile in percentiles:
        cutoff = float(np.percentile(gradient, percentile))
        # Zero derivative is never a barrier, even when its percentile is zero.
        strong = (gradient >= cutoff) & (gradient > 0)
        thresholded_edge = np.where(strong, gradient, 0)
        barrier = strong & ridges if ridge_only else strong.copy()
        if gap_closing_radius_pixels:
            barrier = ndi.binary_closing(barrier, structure=_disk(gap_closing_radius_pixels))
        if removal_mode == "boundary_edges":
            barrier &= domain
            part = remove_boundary_edges(barrier, domain, exterior_band_pixels,
                connectivity, barrier_dilation_pixels, fill_removed_loops)
            part.update(barrier=barrier, thresholded_edge=thresholded_edge)
            details.append(part)
            rows.append(dict(threshold=float(percentile), threshold_mode="edge_percentile",
                method="boundary_edges", intensity_percentile=float(intensity_percentile),
                cutoff_intensity=intensity_cutoff, edge_cutoff=cutoff,
                area_pixels=int(part["mask"].sum()), removed_pixels=int(part["removed"].sum()),
                removed_edge_pixels=int(part["rejected_edges"].sum()),
                removed_enclosed_pixels=int(part["removed_enclosures"].sum()),
                removed_dark_pixels=int((part["removed"] & dark).sum()),
                removed_bright_pixels=int((part["removed"] & ~dark).sum()),
                rejected_edge_components=int(part["rejected_component_count"]),
                kept_edge_components=int(part["kept_component_count"]),
                center_removed=part["center_removed"],
                touches_growth_boundary=bool((part["mask"] & boundary).any()),status=part["status"]))
            continue
        if barrier_dilation_pixels:
            barrier = ndi.binary_dilation(barrier, structure=_disk(barrier_dilation_pixels))
        barrier &= domain
        part = exterior_partition(
            barrier, domain, central, exterior_band_pixels, connectivity, min_area,
            max_area_fraction, require_central_overlap, boundary_margin_pixels)
        part.update(barrier=barrier, thresholded_edge=thresholded_edge,
                    dark_exterior=part["exterior"] & dark,
                    bright_exterior=part["exterior"] & ~dark)
        details.append(part)
        rows.append(dict(
            threshold=float(percentile), threshold_mode="edge_percentile", method="outside_in",
            intensity_percentile=float(intensity_percentile), cutoff_intensity=intensity_cutoff,
            edge_cutoff=cutoff, area_pixels=int(part["mask"].sum()),
            dark_exterior_pixels=int(part["dark_exterior"].sum()),
            bright_exterior_pixels=int(part["bright_exterior"].sum()),
            enclosed_pixels=int(part["enclosed"].sum()),
            retained_components=part["component_count"],
            exterior_seed_pixels=int(part["exterior_seeds"].sum()),
            touches_growth_boundary=bool((part["mask"] & boundary).any()), status=part["status"]))
    return dict(smoothed=smoothed, gradient=gradient, dark=dark,
                intensity_cutoff=intensity_cutoff, growth_region=domain, central_region=central,
                masks=np.asarray([p["mask"] for p in details]), details=details, rows=rows)
