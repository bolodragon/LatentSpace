"""Circle-clipped paired image features, with ID-preserving staged checkpoints."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, rotate
from particle_aggregation import aggregate_sectors, aggregate_polar
from particle_alignment import save_alignment_transforms, apply_saved_alignment

PAIRED_SOURCES = ('cropped_original', 'cropped_derivative',
    'aligned_cropped_original', 'aligned_cropped_derivative',
    'aggregated_original', 'aggregated_derivative', 'fft_original', 'fft_derivative',
    'combined_cropped')


def _stage(root, parent=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='paired_', dir=root))
    manifest = json.loads(json.dumps(parent['settings'])) if parent else dict(files={}, blocks={})
    return folder, manifest


def _save(folder, manifest, key, array, blocks=None):
    path = folder / (key + '.npy')
    np.save(path, array, allow_pickle=False)
    manifest['files'][key] = str(path)
    if blocks is not None: manifest['blocks'][key] = blocks


def _finish(root, folder, manifest, keys):
    (folder/'settings.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for key in keys:
        (Path(root)/('latest_'+key+'.json')).write_text(json.dumps({'directory': str(folder)})+'\n')
    return load_paired_dataset(root, keys[0], folder)


def load_paired_dataset(root, source, directory=None):
    if source not in PAIRED_SOURCES: raise ValueError(f'Unknown paired dataset: {source}')
    root = Path(root).resolve()
    if directory is None:
        pointer = root / ('latest_'+source+'.json')
        if not pointer.exists(): raise FileNotFoundError(f'Run Section 3 to save {source} first')
        directory = json.loads(pointer.read_text())['directory']
    folder = Path(directory)
    if not folder.is_absolute(): folder = root/folder
    settings = json.loads((folder/'settings.json').read_text())
    def read(key):
        if key not in settings['files']: raise ValueError(f'{key} is absent from checkpoint {folder}')
        return np.load(settings['files'][key], mmap_mode='r', allow_pickle=False)
    values = read(source)
    names = read('image_names')
    if len(names) != len(values) or len(set(names.tolist())) != len(names):
        raise ValueError('Dataset IDs must be unique and match the number of rows')
    shape = tuple(settings['image_shape'])
    if source.startswith('aggregated'):
        images = project_bins(values, read('sector_map'))
    elif source == 'combined_cropped': images = read('aligned_cropped_original')
    else: images = values
    return dict(directory=folder, settings=settings, image_names=names, image_shape=shape,
                vectors=values.reshape(len(names), -1), images=images,
                empty_masks=~np.any(read('cropped_derivative') != 0, axis=(1, 2)),
                sector_map=read('sector_map') if 'sector_map' in settings['files'] else None)


def crop_pairs(root, images, names, source_checkpoint, source,
               sigma=1., edge_percentile=70., radius_fraction=.35, preserve_edge_strength=True):
    images = np.asarray(images, dtype=np.float32)
    names = np.asarray(names, dtype=str)
    if images.ndim != 3 or not len(images) or not np.isfinite(images).all():
        raise ValueError('Expected a nonempty finite image stack')
    if len(names) != len(images) or len(set(names.tolist())) != len(names): raise ValueError('Invalid image IDs')
    if not np.isfinite(sigma) or sigma <= 0 or not 0 <= edge_percentile <= 100 or not 0 < radius_fraction <= .5:
        raise ValueError('Require sigma > 0, percentile in [0,100], circle radius fraction in (0,.5]')
    if not isinstance(preserve_edge_strength, (bool, np.bool_)): raise ValueError('preserve_edge_strength must be boolean')
    folder, m = _stage(root)
    shape = images.shape[1:]
    y, x = np.indices(shape)
    circle = (y-(shape[0]-1)/2)**2+(x-(shape[1]-1)/2)**2 <= (min(shape)*radius_fraction)**2
    derivatives = np.empty_like(images)
    cutoffs = []
    for i, image in enumerate(images):
        edge = np.hypot(gaussian_filter(image,sigma,order=(1,0)), gaussian_filter(image,sigma,order=(0,1)))
        cutoff = np.percentile(edge, edge_percentile)
        selected = circle & (edge >= cutoff) & (edge > 0)
        derivatives[i] = np.where(selected, edge, 0) if preserve_edge_strength else selected.astype(np.float32)
        cutoffs.append(cutoff)
    m.update(source_checkpoint=str(Path(source_checkpoint).resolve()), source=source,
             image_shape=list(shape), sigma=sigma, edge_percentile=edge_percentile,
             radius_fraction=radius_fraction, preserve_edge_strength=bool(preserve_edge_strength),
             derivative_representation="magnitude" if preserve_edge_strength else "binary", crop_directory=str(folder))
    for key, array in [('image_names',names),('circle_mask',circle),('edge_cutoffs',cutoffs),
                       ('cropped_original',np.where(circle,images,0)),('cropped_derivative',derivatives)]:
        _save(folder,m,key,array, [dict(name=key,kind='image',start=0,stop=int(np.prod(shape)))] if key.startswith('cropped_') else None)
    return _finish(root,folder,m,['cropped_original','cropped_derivative'])


def _alignment(image, exclusion=15., sector=20., min_axis_ratio=1.1, min_contrast=.05):
    """Major covariance axis; strongest off-axis lobe selects upper/right quadrant."""
    if not 0 < exclusion < 90 or not 0 < sector < 90 or min_axis_ratio < 1 or not 0 <= min_contrast <= 1:
        raise ValueError('Invalid alignment controls')
    center = (np.asarray(image.shape)-1)/2
    y,x = np.indices(image.shape)
    w = np.maximum(image,0).astype(float)
    rec = dict(angle_degrees=0.,aligned_y=center[0],aligned_x=center[1],
               flip_vertical=False,flip_horizontal=False,axis_ratio=1.,secondary_contrast=0.,
               centroid_y=center[0],centroid_x=center[1],axis_y=1.,axis_x=0.,
               secondary_y=0.,secondary_x=0.,secondary_length=0.,status='no_positive_signal')
    if w.sum() <= 0: return rec
    centroid = np.array([(w*y).sum(),(w*x).sum()])/w.sum()
    dy,dx=y-centroid[0],x-centroid[1]
    cov=np.array([[(w*dy*dy).sum(),(w*dy*dx).sum()],[(w*dy*dx).sum(),(w*dx*dx).sum()]])/w.sum()
    vals,vecs=np.linalg.eigh(cov); axis=vecs[:,-1]
    angle=float((-np.degrees(np.arctan2(axis[1],axis[0]))+90)%180-90)
    ratio=float(np.sqrt(max(vals[-1],0)/max(vals[0],1e-12)))
    rotated=rotate(w,angle,reshape=False,order=1,mode='constant',cval=0,prefilter=False)
    t=np.radians(angle); matrix=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    pivot=center+matrix@(centroid-center)
    dy,dx=y-pivot[0],x-pivot[1]
    phi=np.arctan2(dx,-dy)
    energy=rotated*(dy*dy+dx*dx)*(np.abs(np.sin(phi)) >= np.sin(np.radians(exclusion)))
    hist,edges=np.histogram(phi,bins=72,range=(-np.pi,np.pi),weights=energy)
    hist=gaussian_filter1d(hist.astype(float),1,mode='wrap')
    angles=(edges[:-1]+edges[1:])/2
    hist[np.abs(np.sin(angles)) < np.sin(np.radians(exclusion))]=0
    best=int(np.argmax(hist)); direction=np.zeros(2); contrast=0.
    if hist[best]>0:
        # Compare all reflected copies of the chosen direction (ties are ambiguous).
        mirrored=[-angles[best],np.pi-angles[best],angles[best]+np.pi]
        other=[int(np.argmin(np.abs(np.angle(np.exp(1j*(angles-a)))))) for a in mirrored]
        contrast=float(max(0,1-max(hist[j] for j in other)/hist[best]))
        selected=energy*(np.abs(np.angle(np.exp(1j*(phi-angles[best]))))<=np.radians(sector))
        if selected.sum()>0: direction=np.array([(selected*dy).sum(),(selected*dx).sum()])/selected.sum()
    # Always use the deterministic lobe direction, but report unreliable orientation.
    length=float(np.linalg.norm(direction)); original=matrix.T@direction
    rec.update(angle_degrees=angle,flip_vertical=bool(direction[0]>0),flip_horizontal=bool(direction[1]<0),
               centroid_y=centroid[0],centroid_x=centroid[1],axis_y=axis[0],axis_x=axis[1],
               axis_ratio=ratio,secondary_contrast=contrast,secondary_length=length,
               secondary_y=float(original[0]),secondary_x=float(original[1]),
               status='uncertain_axis' if ratio<min_axis_ratio else ('uncertain_secondary' if contrast<min_contrast or length==0 else 'ok'))
    return rec


def align_pairs(root, crop, **controls):
    folder,m=_stage(root,crop)
    original=np.load(m['files']['cropped_original'],mmap_mode='r')
    derivative=np.load(m['files']['cropped_derivative'],mmap_mode='r')
    records=[dict(image_name=str(name),**_alignment(image,**controls)) for name,image in zip(crop['image_names'],derivative)]
    table=pd.DataFrame(records)
    table.to_csv(folder/'alignment.csv',index=False)
    save_alignment_transforms(folder,table,crop['image_shape'])
    m.update(aligned_directory=str(folder),alignment_controls=controls)
    for key,images in [('aligned_cropped_original',original),('aligned_cropped_derivative',derivative)]:
        result,_=apply_saved_alignment(images,crop['image_names'],folder)
        _save(folder,m,key,result,[dict(name=key,kind='image',start=0,stop=int(np.prod(crop['image_shape'])))])
    return _finish(root,folder,m,['aligned_cropped_original','aligned_cropped_derivative'])


def aggregate_pairs(root, aligned, n_sectors=8, method='intensity_sum', start_angle_degrees=45.,
                    n_radial_bands=3, count_mode='fraction', positive_threshold=0.):
    folder,m=_stage(root,aligned)
    n_bins=n_sectors*n_radial_bands
    m.update(n_sectors=n_sectors,n_radial_bands=n_radial_bands,n_bins=n_bins,
             aggregation_method=method,start_angle_degrees=start_angle_degrees,normalization='none',
             count_mode=count_mode,positive_threshold=positive_threshold,
             bin_order='inner to outer equal-width rings, clockwise sectors within each ring')
    for channel in ('original','derivative'):
        images=np.load(m['files']['aligned_cropped_'+channel],mmap_mode='r')
        result=aggregate_polar(images,n_sectors=n_sectors,n_radial_bands=n_radial_bands,
            radius_fraction=m['radius_fraction'],method=method,start_angle_degrees=start_angle_degrees,
            count_mode=count_mode,positive_threshold=positive_threshold)
        key='aggregated_'+channel
        blocks=[dict(name=channel+' '+name,kind='sector',start=i*n_bins,stop=(i+1)*n_bins)
                for i,name in enumerate((method,'positive_pixel_'+count_mode))]
        _save(folder,m,key,result['vectors'],blocks)
        _save(folder,m,'total_positive_pixels_'+channel,result['total_positive_pixels'])
    _save(folder,m,'sector_map',result['sector_map'])
    _save(folder,m,'bin_pixel_counts',result['pixel_counts'])
    return _finish(root,folder,m,['aggregated_original','aggregated_derivative'])


def fft_pairs(root, aligned, remove_support_mean=True):
    folder,m=_stage(root,aligned)
    if not isinstance(remove_support_mean,(bool,np.bool_)):raise ValueError('remove_support_mean must be boolean')
    support=np.load(m['files']['circle_mask']).astype(bool)
    m.update(fft_remove_support_mean=bool(remove_support_mean),
             fft_support='saved centered circle; outside is zero',
             fft_convention='fftshift(fft2(image, norm="ortho")); log1p(abs(spectrum)); no taper')
    for channel in ('original','derivative'):
        images=np.load(m['files']['aligned_cropped_'+channel],mmap_mode='r')
        # Batch FFTs to avoid a large transient complex128 stack.
        complex_path=folder/('fft_complex_'+channel+'.npy')
        spectra=np.lib.format.open_memmap(complex_path,mode='w+',dtype=np.complex64,shape=images.shape)
        feature_path=folder/('fft_'+channel+'.npy')
        logs=np.lib.format.open_memmap(feature_path,mode='w+',dtype=np.float32,shape=images.shape)
        means=np.zeros(len(images),dtype=np.float64)
        for start in range(0,len(images),64):
            batch=np.asarray(images[start:start+64],dtype=np.float64)
            if remove_support_mean:
                means[start:start+len(batch)]=batch[:,support].mean(axis=1)
                batch=np.where(support,batch-means[start:start+len(batch),None,None],0.)
            f=np.fft.fftshift(np.fft.fft2(batch,norm='ortho'),axes=(-2,-1))
            spectra[start:start+64]=f; logs[start:start+64]=np.log1p(np.abs(f))
        spectra.flush(); logs.flush()
        _save(folder,m,'fft_support_mean_'+channel,means)
        m['files']['fft_complex_'+channel]=str(complex_path)
        m['files']['fft_'+channel]=str(feature_path)
        m['blocks']['fft_'+channel]=[dict(name='fft_'+channel,kind='frequency',start=0,stop=int(np.prod(images.shape[1:])))]
    return _finish(root,folder,m,['fft_original','fft_derivative'])


COMBINED_BLOCKS = (
    'aligned_cropped_original', 'aligned_cropped_derivative',
    'original_intensity_sum', 'original_positive_pixel_count',
    'derivative_intensity_sum', 'derivative_positive_pixel_count',
    'fft_original', 'fft_derivative',
)


def combine_pairs(root, aggregated, fft, selected_blocks=None, scaling='none', block_weights=None):
    """Concatenate selected feature blocks in user order; None retains all eight."""
    keys = list(COMBINED_BLOCKS if selected_blocks is None else selected_blocks)
    if not keys or any(not isinstance(k, str) or k not in COMBINED_BLOCKS for k in keys):
        raise ValueError(f'Select at least one block from {COMBINED_BLOCKS}')
    if len(set(keys)) != len(keys):
        raise ValueError('Combined feature blocks must not be repeated')
    if (aggregated['settings']['aligned_directory'] != fft['settings']['aligned_directory']
        or not np.array_equal(aggregated['image_names'],fft['image_names'])):
        raise ValueError('Aggregation and FFT must use exactly the same aligned checkpoint and IDs')
    if scaling not in ('none','block_variance'):raise ValueError('Choose none or block_variance scaling')
    weights=dict(block_weights or {})
    if any(k not in COMBINED_BLOCKS for k in weights):raise ValueError('Unknown block weight key')
    if any(not np.isfinite(v) or v<0 for v in weights.values()):raise ValueError('Weights must be finite and nonnegative')
    if not any(weights.get(k,1.)>0 for k in keys):raise ValueError('At least one selected block must have positive weight')
    folder,m=_stage(root,aggregated)
    m['files'].update({k:v for k,v in fft['settings']['files'].items() if k.startswith('fft_')})
    m['blocks'].update({k:v for k,v in fft['settings']['blocks'].items() if k.startswith('fft_')})
    m['fft_convention']=fft['settings']['fft_convention']
    m['fft_remove_support_mean']=fft['settings'].get('fft_remove_support_mean',False)
    sources = {key: (key, None) for key in COMBINED_BLOCKS if key.startswith(('aligned_', 'fft_'))}
    n = m.get('n_bins',m['n_sectors'])
    for channel in ('original', 'derivative'):
        sources[channel+'_intensity_sum'] = ('aggregated_'+channel, slice(0, n))
        sources[channel+'_positive_pixel_count'] = ('aggregated_'+channel, slice(n, 2*n))
    arrays = []
    for key in keys:
        source, columns = sources[key]
        values = np.load(m['files'][source], mmap_mode='r').reshape(len(aggregated['image_names']), -1)
        arrays.append(values if columns is None else values[:, columns])
    path=folder/'combined_cropped.npy'
    output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(arrays[0]),sum(a.shape[1] for a in arrays)))
    offset=0; blocks=[]; means=[]; gains=[]
    for key,array in zip(keys,arrays):
        source, columns = sources[key]
        block = m['blocks'][source][0 if columns is None or columns.start == 0 else 1]
        # Merge batch means/variance in float64, never scale pixels independently.
        mean=np.zeros(array.shape[1]);m2=np.zeros(array.shape[1]);seen=0
        for start in range(0,len(array),128):
            batch=np.asarray(array[start:start+128],dtype=np.float64)
            bm=batch.mean(axis=0);delta=bm-mean;count=len(batch)
            m2+=((batch-bm)**2).sum(axis=0)+delta**2*seen*count/(seen+count)
            mean+=delta*count/(seen+count);seen+=count
        total_variance=float(m2.sum()/seen)
        scale=np.sqrt(total_variance) if scaling=='block_variance' and total_variance>1e-24 else 1.
        center=mean if scaling=='block_variance' else np.zeros_like(mean)
        gain=float(weights.get(key,1.))/scale
        blocks.append(dict(block, selection_key=key, start=offset, stop=offset+array.shape[1],
                           raw_total_variance=total_variance, scale=float(scale),weight=float(weights.get(key,1.)),
                           gain=gain, constant=total_variance<=1e-24,
                           weighted_total_variance=total_variance*gain**2))
        means.append(center);gains.append(np.full(array.shape[1],gain))
        for start in range(0,len(output),64):
            output[start:start+64,offset:offset+array.shape[1]]=(np.asarray(array[start:start+64],dtype=float)-center)*gain
        offset+=array.shape[1]
    output.flush()
    _save(folder,m,'combined_feature_means',np.concatenate(means))
    _save(folder,m,'combined_feature_gains',np.concatenate(gains))
    m['files']['combined_cropped']=str(path); m['blocks']['combined_cropped']=blocks
    m.update(feature_order=keys,flatten_order='C (row-wise)',feature_scaling=scaling,block_weights=weights,
             scaling_rule='(raw - feature_means) * feature_gains; fitted across images; no per-image normalization',
             aggregated_directory=str(aggregated['directory']),fft_directory=str(fft['directory']))
    return _finish(root,folder,m,['combined_cropped'])


def project_bins(values, bin_map):
    """Map bins to pixels, keeping out-of-support (-1) pixels zero."""
    return np.where(bin_map>=0,values[...,np.maximum(bin_map,0)],0)


def transform_combined(raw_vectors, bundle):
    """Reuse a saved block transform for future images on the same descriptor grid."""
    files=bundle['settings']['files']
    means=np.load(files['combined_feature_means']);gains=np.load(files['combined_feature_gains'])
    values=np.asarray(raw_vectors)
    if values.shape[-1]!=len(means):raise ValueError('Combined feature width differs from the fitted transform')
    return (values-means)*gains


def loading_maps(loadings,bundle,source):
    maps=[]; labels=[]
    for block in bundle['settings']['blocks'][source]:
        values=loadings[:,block['start']:block['stop']]
        maps.append(project_bins(values,bundle['sector_map']) if block['kind']=='sector' else values.reshape(len(values),*bundle['image_shape']))
        labels.append(block['name'])
    return np.stack(maps,axis=1),labels


def preview_pairs(bundle, keys, count=6, dpi=160):
    import matplotlib.pyplot as plt
    m=bundle['settings']; n=min(count,len(bundle['image_names']))
    fig,axes=plt.subplots(n,len(keys),figsize=(3.4*len(keys),3*n),squeeze=False,layout='constrained')
    for col,key in enumerate(keys):
        images=np.load(m['files'][key],mmap_mode='r')
        if images.ndim==2: images=project_bins(images[:n],np.load(m['files']['sector_map']))
        low,high=np.percentile(images[:n],[1,99])
        for row in range(n):
            ax=axes[row,col]; ax.imshow(images[row],cmap='magma' if 'derivative' in key or 'fft' in key else 'gray',vmin=low,vmax=max(high,low+1e-12))
            ax.set_title(key.replace('_',' ')+'\n'+str(bundle['image_names'][row]).split('@')[0]); ax.set_axis_off()
    fig.savefig(bundle['directory']/'preview.png',dpi=dpi,bbox_inches='tight')
    plt.show(); plt.close(fig)


def preview_alignment(bundle, count=6, dpi=160):
    """Overlay the estimated bars before and after the saved transforms."""
    import matplotlib.pyplot as plt
    table=pd.read_csv(Path(bundle['settings']['aligned_directory'])/'alignment.csv')
    images=np.load(bundle['settings']['files']['cropped_derivative'],mmap_mode='r')
    aligned=np.load(bundle['settings']['files']['aligned_cropped_derivative'],mmap_mode='r')
    n=min(count,len(images)); center=(np.asarray(images.shape[1:])-1)/2
    fig,axes=plt.subplots(n,2,figsize=(7,3*n),squeeze=False,layout='constrained')
    for row in range(n):
        rec=table.iloc[row]; origin=np.array([rec.centroid_y,rec.centroid_x])
        major=np.array([rec.axis_y,rec.axis_x])*min(images.shape[1:])*.2
        secondary=np.array([rec.secondary_y,rec.secondary_x])
        points=np.stack([origin,origin-major,origin+major,origin+secondary])
        t=np.radians(rec.angle_degrees)
        rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        transformed=center+(points-center)@rot.T
        if rec.flip_vertical: transformed[:,0]=2*center[0]-transformed[:,0]
        if rec.flip_horizontal: transformed[:,1]=2*center[1]-transformed[:,1]
        for ax,image,p in zip(axes[row],(images[row],aligned[row]),(points,transformed)):
            ax.imshow(image,cmap='gray'); ax.plot(p[1:3,1],p[1:3,0],color='cyan')
            ax.plot(p[[0,3],1],p[[0,3],0],color='orange'); ax.set_axis_off()
        axes[row,0].set_title(str(rec.image_name).split('@')[0]+' | '+rec.status)
        axes[row,1].set_title(f'{rec.angle_degrees:.1f}° | flip up/down {rec.flip_vertical} | left/right {rec.flip_horizontal}')
    fig.savefig(bundle['directory']/'alignment_bars.png',dpi=dpi,bbox_inches='tight')
    plt.show(); plt.close(fig)
