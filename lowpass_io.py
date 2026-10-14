"""Batched Gaussian low-pass filtering and reusable CWF input checkpoints."""
import json
import os
import tempfile
from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift


def star_recentering_plan(metadata, optics, image_shape, pixel_size):
    """RELION origins translate observations to reference: +origin/pixel_size.

    References: RELION conventions and particle_subtractor.cpp selfTranslate.
    No rotations/CTF operations are applied; keep those fields for CWF.
    """
    if 'star_recentering_applied' in metadata and metadata['star_recentering_applied'].fillna(False).any():
        raise ValueError('These metadata already describe recentered images; do not apply shifts twice')
    required=['rlnOpticsGroup','rlnOriginXAngst','rlnOriginYAngst']
    if not all(c in metadata for c in required):
        raise ValueError('Recentering requires optics group and both STAR origins in Angstroms')
    by_group=optics.set_index('rlnOpticsGroup')
    if not by_group.index.is_unique:
        raise ValueError('Optics group IDs must be unique')
    sampling=metadata['rlnOpticsGroup'].map(by_group['rlnImagePixelSize']).to_numpy(dtype=float)
    if not np.isfinite(sampling).all() or np.any(sampling<=0) or not np.allclose(sampling,pixel_size,rtol=1e-3):
        raise ValueError('STAR optics sampling must match the source MRC pixel size')
    boxes=metadata['rlnOpticsGroup'].map(by_group['rlnImageSize']).to_numpy(dtype=float)
    if len(image_shape)!=2 or image_shape[0]!=image_shape[1] or not np.all(boxes==image_shape[0]):
        raise ValueError('STAR optics box size does not match particle image dimensions')
    offsets=metadata[['rlnOriginXAngst','rlnOriginYAngst']].to_numpy(dtype=float)
    if not np.isfinite(offsets).all():
        raise ValueError('Non-finite STAR origins')
    shifts_xy=offsets/sampling[:,None]
    if np.any(np.abs(shifts_xy)>=np.array(image_shape[::-1])/2):
        raise ValueError('Origin exceeds half the image size; check STAR/stack pairing')
    corrected=metadata.copy()
    for axis,j in [('X',0),('Y',1)]:
        field=f'rlnOrigin{axis}Angst'
        corrected[f'original_{field}']=corrected[field]
        corrected[f'applied_shift_{axis.lower()}_pixels']=shifts_xy[:,j]
        corrected[field]=0.0
    corrected['star_recentering_applied']=True
    return shifts_xy, corrected


def recenter_star_images(images, shifts_xy):
    """Subpixel translation; input x/y offsets map to array (y,x) axis order."""
    images=np.asarray(images,dtype=np.float32)
    shifts_xy=np.asarray(shifts_xy,dtype=float)
    if images.ndim!=3 or shifts_xy.shape!=(len(images),2) or not np.isfinite(shifts_xy).all() or not np.isfinite(images).all():
        raise ValueError('Need finite images and one finite (x,y) shift per image')
    output=np.empty_like(images)
    for i,(dx,dy) in enumerate(shifts_xy):
        output[i]=shift(images[i],shift=(dy,dx),order=3,mode='reflect',prefilter=True)
    return output


def smooth_lowpass(images, sigma_pixels=1.0):
    """Filter only spatial axes; sigma is in original image pixels, not Angstroms."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3 or not np.isfinite(images).all():
        raise ValueError('Expected finite images with shape (N, height, width)')
    if not np.isfinite(sigma_pixels) or sigma_pixels <= 0:
        raise ValueError('sigma_pixels must be positive and finite')
    return gaussian_filter(images, sigma=(0, sigma_pixels, sigma_pixels), mode='reflect')


def save_lowpass_inputs(particles, noise, metadata, optics, root, pixel_size,
                        sigma_pixels=1.0, batch_size=32, source_paths=None, recenter_from_star=False):
    """Apply the identical filter to both stacks and stream outputs to a new folder."""
    if (particles.ndim != 3 or noise.ndim != 3 or particles.shape[1:] != noise.shape[1:]
            or min(len(particles),len(noise)) < 2 or len(metadata) != len(particles)):
        raise ValueError('Need matching particle/noise stacks and one metadata row per particle')
    if batch_size < 1 or not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('Invalid batch size or pixel size')
    if not np.isfinite(sigma_pixels) or sigma_pixels <= 0:
        raise ValueError('sigma_pixels must be positive and finite')
    shifts_xy=corrected_metadata=noise_shift_rows=None
    if recenter_from_star:
        shifts_xy,corrected_metadata=star_recentering_plan(metadata,optics,particles.shape[1:],pixel_size)
        noise_shift_rows=np.arange(len(noise)) % len(particles)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='lowpass_',dir=root))
    # The particle NPY can be reopened as a memory map; noise MRC feeds CWF PSD estimation directly.
    target=np.lib.format.open_memmap(folder/'particles.npy',mode='w+',dtype=np.float32,shape=particles.shape)
    for start in range(0,len(particles),batch_size):
        stop=min(start+batch_size,len(particles))
        block=particles[start:stop]
        if recenter_from_star:
            block=recenter_star_images(block,shifts_xy[start:stop])
        target[start:stop]=smooth_lowpass(block,sigma_pixels)
    target.flush();del target
    with mrcfile.new_mmap(str(folder/'noise.mrc'),shape=noise.shape,mrc_mode=2,overwrite=False) as target:
        for start in range(0,len(noise),batch_size):
            stop=min(start+batch_size,len(noise))
            block=noise[start:stop]
            if recenter_from_star:
                block=recenter_star_images(block,shifts_xy[noise_shift_rows[start:stop]])
            target.data[start:stop]=smooth_lowpass(block,sigma_pixels)
        target.set_image_stack();target.voxel_size=float(pixel_size)
        target.update_header_stats();target.flush()
    (corrected_metadata if recenter_from_star else metadata).to_csv(folder/'particles.csv',index=False)
    if recenter_from_star:
        metadata.to_csv(folder/'particles_original.csv',index=False)
        pd.DataFrame({'noise_row_1based':np.arange(1,len(noise)+1),
                      'assigned_particle_row_1based':noise_shift_rows+1,
                      'assigned_particle_image_name':metadata['rlnImageName'].to_numpy()[noise_shift_rows],
                      'applied_shift_x_pixels':shifts_xy[noise_shift_rows,0],
                      'applied_shift_y_pixels':shifts_xy[noise_shift_rows,1]}).to_csv(folder/'noise_transforms.csv',index=False)
    optics.to_csv(folder/'optics.csv',index=False)
    settings={'method':'Gaussian spatial low-pass','sigma_pixels':float(sigma_pixels),
              'sigma_angstrom':float(sigma_pixels*pixel_size),'boundary':'reflect',
              'pixel_size_angstrom':float(pixel_size),'particle_shape':list(particles.shape),
              'noise_shape':list(noise.shape),'sources':source_paths or {},'schema_version':2,
              'star_recentering':bool(recenter_from_star),
              'translation_convention':'+(OriginYAngst, OriginXAngst)/pixel_size in array axes' if recenter_from_star else None,
              'translation_interpolation':'cubic order=3, reflect boundary' if recenter_from_star else None,
              'noise_translation':'particle offsets assigned cyclically as preprocessing control; not fitted noise alignment' if recenter_from_star else None,
              'rotations_applied':False,'ctf_correction_applied':False}
    (folder/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    pointer=root/'latest_lowpass.json';temp=pointer.with_suffix('.tmp')
    temp.write_text(json.dumps({'directory':folder.name},indent=2)+'\n');os.replace(temp,pointer)
    return folder


def load_lowpass_inputs(root, directory=None):
    root=Path(root)
    if directory is None:
        pointer=root/'latest_lowpass.json'
        if not pointer.exists():
            raise FileNotFoundError('No low-pass checkpoint; run section 1.4 once first')
        folder=root/json.loads(pointer.read_text())['directory']
    else:
        folder=Path(directory)
        if not folder.is_absolute():folder=root/folder
    settings=json.loads((folder/'settings.json').read_text())
    particles=np.load(folder/'particles.npy',mmap_mode='r',allow_pickle=False)
    with mrcfile.mmap(str(folder/'noise.mrc'),mode='r') as mrc:
        noise=mrc.data.copy()
        if not np.isclose(float(mrc.voxel_size.x),settings['pixel_size_angstrom']):
            raise ValueError('Noise sampling disagrees with saved settings')
    metadata=pd.read_csv(folder/'particles.csv',dtype={'rlnImageName':str})
    optics=pd.read_csv(folder/'optics.csv')
    if (list(particles.shape)!=settings['particle_shape'] or list(noise.shape)!=settings['noise_shape']
            or len(metadata)!=len(particles)):
        raise ValueError('Low-pass checkpoint shapes/metadata disagree')
    return {'particles':particles,'noise':noise,'metadata':metadata,'optics':optics,
            'noise_path':folder/'noise.mrc','directory':folder,'settings':settings}
