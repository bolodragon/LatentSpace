"""Clockwise radial-sector descriptors for aligned images."""
import json
from pathlib import Path
import numpy as np

# Extend this registry to add a new scoring method. Each receives (N, pixels).
SECTOR_SCORERS = {'intensity_sum': lambda pixels: pixels.sum(axis=1, dtype=np.float64)}


def aggregate_sectors(images, n_sectors=8, method='intensity_sum', normalization='none',
                      start_angle_degrees=0.0):
    """Score sectors starting at an angle clockwise from 12 o'clock."""
    images = np.asarray(images)
    if images.ndim != 3 or not len(images) or not np.isfinite(images).all():
        raise ValueError('Expected a nonempty finite image stack')
    if isinstance(n_sectors,(bool,np.bool_)) or not isinstance(n_sectors,(int,np.integer)) or n_sectors < 1:
        raise ValueError('n_sectors must be a positive integer')
    if method not in SECTOR_SCORERS: raise ValueError(f'Unknown scorer; choose from {list(SECTOR_SCORERS)}')
    if normalization not in ('l2','l1','none'): raise ValueError('Choose l2, l1, or none normalization')
    start_angle_degrees = float(start_angle_degrees)
    if not np.isfinite(start_angle_degrees):
        raise ValueError("start_angle_degrees must be finite")
    start_angle_degrees %= 360.0
    h,w=images.shape[1:];yy,xx=np.indices((h,w))
    dy=yy-(h-1)/2;dx=xx-(w-1)/2
    angle=np.mod(np.arctan2(dx,-dy)-np.deg2rad(start_angle_degrees),2*np.pi)
    angle[(dx==0)&(dy==0)]=0  # Odd-grid center belongs deterministically to sector 1.
    sector_map=np.minimum((angle*n_sectors/(2*np.pi)).astype(int),n_sectors-1)
    counts=np.bincount(sector_map.ravel(),minlength=n_sectors)
    if np.any(counts==0): raise ValueError('Too many sectors for this pixel grid: some sectors are empty')
    scores=np.column_stack([SECTOR_SCORERS[method](images[:,sector_map==j]) for j in range(n_sectors)])
    positive_counts = np.column_stack([np.count_nonzero(images[:,sector_map==j] > 0, axis=1)
                                       for j in range(n_sectors)])
    scores = np.concatenate([scores, positive_counts], axis=1)
    if not np.isfinite(scores).all(): raise ValueError('Non-finite sector scores')
    scale=(np.linalg.norm(scores,axis=1) if normalization=='l2' else
           np.abs(scores).sum(axis=1) if normalization=='l1' else np.ones(len(scores)))
    zero=np.all(scores==0,axis=1)
    vectors=np.divide(scores,scale[:,None],out=np.zeros_like(scores),where=~zero[:,None])
    return dict(vectors=vectors.astype(np.float32),raw_scores=scores,normalization_scale=scale,
                zero_vectors=zero,sector_map=sector_map,pixel_counts=counts,
                positive_pixel_counts=positive_counts)


def load_aggregated_results(root,directory=None):
    root=Path(root).resolve()
    if directory is None:
        pointer=root/'latest_aggregated.json'
        if not pointer.exists(): raise FileNotFoundError('Run Section 3.8 first')
        directory=json.loads(pointer.read_text())['directory']
    folder=Path(directory)
    if not folder.is_absolute():folder=root/folder
    settings=json.loads((folder/'settings.json').read_text())
    with np.load(folder/'aggregated.npz',allow_pickle=False) as f:
        result={k:f[k].copy() for k in f.files}
    if result['vectors'].shape != (len(result['image_names']),settings.get('n_features', settings['n_sectors'])):
        raise ValueError('Aggregated vectors and IDs disagree')
    return dict(result,directory=folder,settings=settings,image_shape=result['sector_map'].shape)


def aggregate_polar(images, n_sectors=8, n_radial_bands=3, radius_fraction=.35,
                    method='intensity_sum', start_angle_degrees=0., count_mode='fraction',
                    positive_threshold=0.):
    """Ring-major, clockwise bins restricted to a centered circle.

    Return raw intensity sums, plus counts or occupancy fractions. This is not
    normalization across the feature vector; fractions divide by each bin's area.
    """
    images=np.asarray(images)
    if images.ndim!=3 or not len(images) or not np.isfinite(images).all():
        raise ValueError('Expected a finite nonempty image stack')
    for value in (n_sectors,n_radial_bands):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError('Sectors and radial bands must be positive integers')
    if not 0<radius_fraction<=.5 or not np.isfinite(start_angle_degrees) or not np.isfinite(positive_threshold):
        raise ValueError('Invalid radius, first-cut angle, or positive threshold')
    if method not in SECTOR_SCORERS or count_mode not in ('count','fraction'):
        raise ValueError('Unknown scoring method or count_mode')
    h,w=images.shape[1:];y,x=np.indices((h,w));dy=y-(h-1)/2;dx=x-(w-1)/2
    radius=min(h,w)*radius_fraction;r=np.hypot(dx,dy);support=r<=radius
    phi=np.mod(np.arctan2(dx,-dy)-np.deg2rad(start_angle_degrees%360),2*np.pi)
    phi[(dx==0)&(dy==0)]=0
    sectors=np.minimum((phi*n_sectors/(2*np.pi)).astype(int),n_sectors-1)
    rings=np.minimum((r/radius*n_radial_bands).astype(int),n_radial_bands-1)
    bins=np.where(support,rings*n_sectors+sectors,-1)
    size=n_sectors*n_radial_bands
    areas=np.bincount(bins[support],minlength=size)
    if np.any(areas==0):raise ValueError('Some radial/sector bins are empty; reduce bands or sectors')
    sums=np.empty((len(images),size),dtype=np.float64);counts=np.empty_like(sums)
    for j in range(size):
        pixels=images[:,bins==j]
        sums[:,j]=SECTOR_SCORERS[method](pixels)
        counts[:,j]=np.count_nonzero(pixels>positive_threshold,axis=1)
    occupancy=counts/areas if count_mode=='fraction' else counts
    return dict(vectors=np.concatenate([sums,occupancy],axis=1).astype(np.float32),
                sector_map=bins,pixel_counts=areas,positive_pixel_counts=counts,
                total_positive_pixels=counts.sum(axis=1),support_mask=support)
