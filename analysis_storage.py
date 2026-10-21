"""Portable numeric checkpoints for PCA and named clustering results (no pickle)."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from particle_paired import PAIRED_SOURCES, load_paired_dataset
from clustering_results import register_clustering_result


def _json(value):
    if isinstance(value, Path): return str(value.resolve())
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json)+'\n')


def _folder(root, prefix):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix,dir=root))


def resolve(root, directory, pointer):
    root=Path(root).resolve()
    if directory is None:
        path=root/pointer
        if not path.exists(): raise FileNotFoundError(f'No saved checkpoint: {path}. Run the producing section first.')
        directory=json.loads(path.read_text())['directory']
    directory=Path(directory)
    return directory if directory.is_absolute() else root/directory


def load_input_bundle(root, source, directory):
    if source in PAIRED_SOURCES:return load_paired_dataset(root,source,directory)
    if source=='aligned':
        from particle_alignment import load_aligned_results
        return load_aligned_results(root,directory)
    if source=='aggregated':
        from particle_aggregation import load_aggregated_results
        return load_aggregated_results(root,directory)
    if source=='fft_circle':
        from particle_fft import load_fft_results
        return load_fft_results(root,directory)
    if source in ('rotated','rotated_average'):
        from particle_segmentation import load_section3_results
        return load_section3_results(root,directory)
    return None


def _save_table(folder,table):
    # JSON table schema preserves integer labels and string IDs, including leading zeros.
    table.to_json(folder/'table.json',orient='table',index=False,double_precision=15)
    table.to_csv(folder/'table.csv',index=False)


def save_pca(root, namespace):
    ns=namespace;folder=_folder(root,'pca_');model=ns['cwf_pca']
    attrs=['components_','mean_','explained_variance_','explained_variance_ratio_',
           'singular_values_','n_components_','n_samples_','n_features_in_','noise_variance_']
    np.savez_compressed(folder/'model.npz',**{k:getattr(model,k) for k in attrs})
    np.save(folder/'scores.npy',ns['cwf_pca_embedding'])
    np.savez_compressed(folder/'display.npz',circle_mask=ns['cwf_circle_mask'],
        denoised=ns['cwf_display_pixels']['denoised'],noise_denoised=ns['cwf_display_pixels']['noise_denoised'],
        noise_indices=ns['cwf_pca_noise_indices'])
    _save_table(folder,ns['cwf_pca_table'])
    settings=dict(schema_version=1,input_source=ns['cwf_pca_source'],input_directory=str(Path(ns['cwf_loaded_directory']).resolve()),
        input_shape=list(ns['cwf_pca_input_shape']),seed=ns['cwf_pca_seed'],
        display_description=ns['cwf_display_description'],whiten=bool(model.whiten))
    write_json(folder/'settings.json',settings)
    write_json(Path(root)/'latest_pca.json',dict(directory=str(folder)))
    return folder


def restore_pca(namespace,root,directory=None):
    folder=resolve(root,directory,'latest_pca.json');settings=json.loads((folder/'settings.json').read_text())
    if settings['schema_version']!=1:raise ValueError('Unsupported PCA checkpoint version')
    model=PCA(whiten=settings['whiten'])
    with np.load(folder/'model.npz',allow_pickle=False) as f:
        for key in f.files:
            value=f[key];setattr(model,key,value.item() if value.ndim==0 else value.copy())
    scores=np.load(folder/'scores.npy',allow_pickle=False)
    table=pd.read_json(folder/'table.json',orient='table')
    if len(scores)!=len(table) or scores.shape[1]!=model.n_components_:raise ValueError('PCA checkpoint dimensions disagree')
    with np.load(folder/'display.npz',allow_pickle=False) as f:
        pixels={k:f[k].copy() for k in ('denoised','noise_denoised')}
        circle=f['circle_mask'].copy();noise=f['noise_indices'].copy()
    source=settings['input_source'];directory=Path(settings['input_directory'])
    namespace.update(cwf_pca=model,cwf_pca_embedding=scores,cwf_pca_table=table,
        cwf_pca_source=source,cwf_loaded_directory=directory,
        cwf_section3=load_input_bundle(root,source,directory),
        cwf_pca_input_shape=tuple(settings['input_shape']),cwf_pca_seed=settings['seed'],
        cwf_pca_components=int(model.n_components_),cwf_pca_noise_indices=noise,
        cwf_display_pixels=pixels,cwf_circle_mask=circle,
        cwf_display_description=settings['display_description'],cwf_pca_checkpoint_directory=folder)
    return folder


def ensure_pca(namespace,root,directory=None):
    if directory is not None or 'cwf_pca_embedding' not in namespace:
        folder=restore_pca(namespace,root,directory)
        print(f'Restored PCA: {folder}')
    elif 'cwf_pca_checkpoint_directory' not in namespace:
        namespace['cwf_pca_checkpoint_directory']=save_pca(root,namespace)
        print(f"Saved existing PCA: {namespace['cwf_pca_checkpoint_directory']}")


def save_clustering(root,name,result,*,arrays=None,parent_pca=None):
    folder=_folder(root,'clustering_')
    _save_table(folder,result['table'])
    np.savez_compressed(folder/'arrays.npz',**(arrays or {}))
    # Preserve the small lineage settings even for a future custom representation.
    bundle=result.get('input_bundle')
    settings=dict(schema_version=1,name=name,input_source=result['input_source'],
        input_directory=str(Path(result['input_directory']).resolve()),parameters=result['parameters'],
        parent_pca=str(parent_pca) if parent_pca is not None else None,
        input_bundle_settings=bundle.get('settings') if bundle is not None else None)
    write_json(folder/'settings.json',settings)
    index=Path(root)/'clustering_index.json'
    entries=json.loads(index.read_text()) if index.exists() else {}
    entries[name]=str(folder);write_json(index,entries)
    result['checkpoint_directory']=folder
    return folder


def load_clustering(root,name,directory=None):
    root=Path(root).resolve()
    if directory is None:
        index=root/'clustering_index.json';entries=json.loads(index.read_text()) if index.exists() else {}
        if name not in entries:raise FileNotFoundError(f'No saved result {name!r}. Saved names: {list(entries)}')
        directory=entries[name]
    folder=Path(directory)
    if not folder.is_absolute():folder=root/folder
    settings=json.loads((folder/'settings.json').read_text())
    if settings['schema_version']!=1 or settings['name']!=name:raise ValueError('Clustering checkpoint name/version mismatch')
    table=pd.read_json(folder/'table.json',orient='table')
    bundle=load_input_bundle(root,settings['input_source'],settings['input_directory'])
    if bundle is None and settings.get('input_bundle_settings') is not None:
        bundle={'settings':settings['input_bundle_settings']}
    registry={}
    result=register_clustering_result(registry,name,table,input_source=settings['input_source'],
        input_directory=Path(settings['input_directory']),input_bundle=bundle,parameters=settings['parameters'])
    result['checkpoint_directory']=folder
    result['parent_pca']=settings.get('parent_pca')
    with np.load(folder/'arrays.npz',allow_pickle=False) as f:result['arrays']={k:f[k].copy() for k in f.files}
    return result


def save_figure(folder,name,figure):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    figure.savefig(folder/name,dpi=160,bbox_inches='tight')
