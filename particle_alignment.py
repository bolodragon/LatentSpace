"""Intensity-weighted principal-axis alignment of masked particle images."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import rotate, affine_transform


def align_vertical(image, min_axis_ratio=1.1, min_direction_skew=0.05,
                   secondary_exclusion_degrees=15., secondary_sector_degrees=20.,
                   min_secondary_contrast=0.05):
    """Rotate the major axis, then use a nonorthogonal lobe direction to reflect.

    The secondary ray points at the strongest off-axis radial-intensity sector.
    It is a shape landmark rather than the second covariance eigenvector.
    """
    from scipy.ndimage import gaussian_filter1d
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('Expected a finite 2D image')
    if not np.isfinite(min_axis_ratio) or min_axis_ratio < 1:
        raise ValueError('min_axis_ratio must be finite and >= 1')
    if not np.isfinite(min_direction_skew) or min_direction_skew < 0:
        raise ValueError('min_direction_skew must be finite and >= 0')
    if (not 0 < secondary_exclusion_degrees < 90 or not 0 < secondary_sector_degrees < 90
            or not 0 <= min_secondary_contrast <= 1):
        raise ValueError('Secondary angles must be in (0, 90); contrast in [0, 1]')
    w = np.maximum(image.astype(float), 0)
    center = (np.asarray(image.shape)-1)/2
    total = w.sum()
    record = dict(angle_degrees=0., centroid_y=center[0], centroid_x=center[1],
        aligned_y=center[0], aligned_x=center[1], axis_y=1., axis_x=0.,
        bar_half_length=0., secondary_y=0., secondary_x=0., secondary_half_length=0.,
        secondary_aligned_y=0., secondary_aligned_x=0., secondary_contrast=0.,
        axis_ratio=1., major_skew=0., flip_vertical=False, flip_horizontal=False,
        major_direction_resolved=False, secondary_direction_resolved=False, status='no_positive_signal')
    if total <= 0: return image.copy(), record
    yy, xx = np.indices(image.shape)
    centroid = np.array([(w*yy).sum(),(w*xx).sum()])/total
    dy, dx = yy-centroid[0], xx-centroid[1]
    cov = np.array([[(w*dy*dy).sum(),(w*dy*dx).sum()],
                    [(w*dy*dx).sum(),(w*dx*dx).sum()]])/total
    eig, vec = np.linalg.eigh(cov)
    axis = vec[:, -1]
    angle = float((-np.degrees(np.arctan2(axis[1],axis[0]))+90)%180-90)
    ratio = float(np.sqrt(max(eig[-1],0)/max(eig[0],1e-12)))
    if eig[-1] <= 1e-12: angle, ratio = 0., 1.
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    aligned_center = center+rot@(centroid-center)
    output = rotate(image,angle,reshape=False,order=1,mode='constant',cval=0.,prefilter=False)
    # Resolve the main axis direction first, independently of the side landmark.
    rw = np.maximum(output.astype(float),0)
    mass = rw.sum()
    delta = yy-aligned_center[0]
    variance = (rw*delta**2).sum()/mass if mass > 0 else 0.
    skew = float((rw*delta**3).sum()/mass/variance**1.5) if variance > 1e-12 else 0.
    flip_v = skew < -min_direction_skew
    if flip_v:
        output = affine_transform(output,np.diag([-1,1]),offset=aligned_center*np.array([2,0]),
                                  order=1,mode='constant',cval=0.,prefilter=False)
    # Find a prominent off-axis arm/lobe using angular radial energy. Exclude
    # cones around both ends of the primary axis, not a fixed perpendicular ray.
    rw = np.maximum(output.astype(float),0)
    dy, dx = yy-aligned_center[0], xx-aligned_center[1]
    r2 = dy*dy+dx*dx
    phi = np.arctan2(dx,dy)
    offaxis = np.abs(np.sin(phi)) >= np.sin(np.radians(secondary_exclusion_degrees))
    energy = rw*r2*offaxis
    bins = 72
    histogram, edges = np.histogram(phi,bins=bins,range=(-np.pi,np.pi),weights=energy)
    smooth = gaussian_filter1d(histogram.astype(float),1.,mode='wrap')
    centers = (edges[:-1]+edges[1:])/2
    smooth[np.abs(np.sin(centers)) < np.sin(np.radians(secondary_exclusion_degrees))] = 0
    best = int(np.argmax(smooth))
    direction = np.zeros(2)
    contrast = 0.
    resolved = False
    if smooth[best] > 0:
        # Compare against the mirrored sector: a symmetric pair has no preferred side.
        mirror = bins-1-best
        contrast = float((smooth[best]-smooth[mirror])/smooth[best])
        difference = np.abs(np.angle(np.exp(1j*(phi-centers[best]))))
        sector_w = energy*(difference <= np.radians(secondary_sector_degrees))
        if sector_w.sum() > 0:
            direction = np.array([(sector_w*dy).sum(),(sector_w*dx).sum()])/sector_w.sum()
            resolved = bool(contrast > min_secondary_contrast and abs(direction[1]) > 1e-6)
    length = float(np.linalg.norm(direction))
    original_direction = rot.T @ (direction*np.array([-1 if flip_v else 1,1]))
    flip_h = bool(resolved and direction[1] < 0)
    if flip_h:
        output = affine_transform(output,np.diag([1,-1]),offset=aligned_center*np.array([0,2]),
                                  order=1,mode='constant',cval=0.,prefilter=False)
        direction[1] *= -1
    record.update(angle_degrees=angle,centroid_y=centroid[0],centroid_x=centroid[1],
        aligned_y=aligned_center[0],aligned_x=aligned_center[1],axis_y=axis[0],axis_x=axis[1],
        bar_half_length=float(2*np.sqrt(max(eig[-1],0))),
        secondary_y=float(original_direction[0]/length) if length else 0.,
        secondary_x=float(original_direction[1]/length) if length else 0.,
        secondary_half_length=length,
        secondary_aligned_y=float(direction[0]),secondary_aligned_x=float(direction[1]),
        secondary_contrast=contrast,axis_ratio=ratio,major_skew=skew,
        flip_vertical=bool(flip_v),flip_horizontal=flip_h,
        major_direction_resolved=bool(abs(skew)>min_direction_skew),secondary_direction_resolved=resolved,
        status='ok' if ratio>=min_axis_ratio else 'uncertain_axis')
    return output,record


def load_aligned_results(root, directory=None):
    root = Path(root).resolve()
    if directory is None:
        pointer = root/'latest_aligned.json'
        if not pointer.exists():
            raise FileNotFoundError('Run Section 3.7 to save aligned images first')
        directory = json.loads(pointer.read_text())['directory']
    folder = Path(directory)
    if not folder.is_absolute(): folder = root/folder
    settings = json.loads((folder/'settings.json').read_text())
    images = np.load(folder/'aligned_images.npy',mmap_mode='r',allow_pickle=False)
    with np.load(folder/'metadata.npz',allow_pickle=False) as meta:
        names = meta['image_names'].copy()
        empty = meta['empty_masks'].copy()
    if images.ndim != 3 or len(images)!=len(names) or empty.shape!=(len(names),):
        raise ValueError('Aligned image dimensions and IDs disagree')
    return dict(directory=folder,images=images,image_names=names,empty_masks=empty,
                transforms_path=folder/"transforms.npz",
                image_shape=images.shape[1:],settings=settings)


def save_alignment_transforms(directory, table, image_shape):
    """Store the exact sequential operations, pivots, and particle IDs for replay."""
    folder = Path(directory)
    np.savez_compressed(folder/'transforms.npz',
        image_names=table['image_name'].to_numpy(dtype=str),
        angles_degrees=table['angle_degrees'].to_numpy(dtype=float),
        reflection_centers_yx=table[['aligned_y','aligned_x']].to_numpy(dtype=float),
        flip_vertical=table['flip_vertical'].to_numpy(dtype=bool),
        flip_horizontal=table['flip_horizontal'].to_numpy(dtype=bool),
        image_shape=np.asarray(image_shape,dtype=int), schema_version=1,
        operation_order='rotate, vertical_reflection, horizontal_reflection',
        interpolation_order=1, boundary_mode='constant', fill_value=0., reshape=False)
    return folder/'transforms.npz'


def apply_saved_alignment(images, image_names, directory):
    """Replay saved transforms by full ID; no segmentation or axis estimation.

    Require the same pixel grid. Reflections use the stored post-rotation centroid,
    not the centroid of the newly selected images. Each operation is resampled
    separately, exactly as in align_vertical.
    """
    folder = Path(directory)
    images = np.asarray(images,dtype=np.float32)
    with np.load(folder/'transforms.npz',allow_pickle=False) as t:
        if int(t['schema_version']) != 1:
            raise ValueError('Unsupported alignment transform version')
        shape = tuple(t['image_shape'])
        if images.ndim != 3 or images.shape[1:] != shape or len(images)!=len(image_names):
            raise ValueError(f'Transforms require images on the same {shape} pixel grid and matching IDs')
        if not np.isfinite(images).all(): raise ValueError('Images contain non-finite pixels')
        names = t['image_names'].astype(str).tolist()
        if len(set(names))!=len(names): raise ValueError('Duplicate transform image IDs')
        lookup = {name:i for i,name in enumerate(names)}
        try: rows = np.asarray([lookup[str(name)] for name in image_names],dtype=int)
        except KeyError as error:
            raise ValueError(f'No saved transform for image ID {error.args[0]}') from error
        angles=t['angles_degrees'][rows]
        centers=t['reflection_centers_yx'][rows]
        fv=t['flip_vertical'][rows];fh=t['flip_horizontal'][rows]
    output=np.empty_like(images)
    for row,image in enumerate(images):
        result=rotate(image,float(angles[row]),reshape=False,order=1,mode='constant',cval=0.,prefilter=False)
        if fv[row]:
            result=affine_transform(result,np.diag([-1,1]),offset=centers[row]*[2,0],
                                    order=1,mode='constant',cval=0.,prefilter=False)
        if fh[row]:
            result=affine_transform(result,np.diag([1,-1]),offset=centers[row]*[0,2],
                                    order=1,mode='constant',cval=0.,prefilter=False)
        output[row]=result
    return output, dict(angles_degrees=angles,flip_vertical=fv,flip_horizontal=fh,
                        reflection_centers_yx=centers)
