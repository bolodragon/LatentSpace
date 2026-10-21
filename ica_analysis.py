"""ID-preserving ICA representations for interchangeable clustering methods."""
from pathlib import Path
import json
import tempfile
import warnings
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def register_ica_input(registry, name, values, table, *, input_source,
                       input_directory, input_bundle=None, feature_names=None):
    values = np.asarray(values, dtype=float)
    required = ['source', 'image_name', 'source_row_1based']
    if not isinstance(name, str) or not name or any(c not in table for c in required):
        raise ValueError('Provide a name and source/image_name/source_row_1based metadata')
    if (values.ndim != 2 or len(values) != len(table) or not len(values)
            or not np.isfinite(values).all() or table[required].isna().any().any()
            or table.duplicated(['source', 'image_name']).any()):
        raise ValueError('ICA inputs must be finite, with one unique source/image ID per row')
    names = list(feature_names) if feature_names is not None else [f'Feature{i+1}' for i in range(values.shape[1])]
    if len(names) != values.shape[1]: raise ValueError('Feature names must match the input columns')
    registry[name] = dict(values=values.copy(), table=table.copy(deep=True), feature_names=names,
        input_source=input_source, input_directory=input_directory, input_bundle=input_bundle)
    return registry[name]


def select_columns(selection, count, label='columns'):
    chosen = list(range(1, count+1)) if selection is None else list(selection)
    if (not chosen or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                          or not 1 <= i <= count for i in chosen) or len(set(chosen)) != len(chosen)):
        raise ValueError(f'Select distinct 1-based {label} between 1 and {count}')
    return [int(i) for i in chosen]


def fit_ica(bundle, columns=None, n_components=None, *, seed=0, max_iter=2000,
            tol=1e-4, fun='logcosh', threads=2):
    selected = select_columns(columns, bundle['values'].shape[1], 'input columns')
    values = bundle['values'][:, np.asarray(selected)-1]
    rank = int(np.linalg.matrix_rank(values-values.mean(axis=0)))
    n = len(selected) if n_components is None else n_components
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 1 <= n <= min(rank, len(values)-1):
        raise ValueError(f'n_components must be 1..{min(rank,len(values)-1)} (centered input rank); select fewer components/columns')
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('Use a positive integer max_iter and positive finite tolerance')
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('Threads must be a positive integer')
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=threads):
        warnings.simplefilter('always', ConvergenceWarning)
        model = FastICA(n_components=int(n), algorithm='parallel', whiten='unit-variance',
                        fun=fun, max_iter=max_iter, tol=tol, random_state=seed)
        scores = model.fit_transform(values)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if not np.isfinite(scores).all(): raise ValueError('ICA produced nonfinite scores')
    table = bundle['table'][['source','image_name','source_row_1based']].copy(deep=True)
    for i in range(scores.shape[1]): table[f'IC{i+1}'] = scores[:, i]
    settings = dict(input_total_columns=bundle["values"].shape[1], input_columns=selected, input_feature_names=[bundle['feature_names'][i-1] for i in selected],
                    n_components=int(n), seed=seed, max_iter=max_iter, tol=tol, fun=fun,
                    whiten='unit-variance', converged=converged, iterations=int(model.n_iter_),
                    input_source=bundle['input_source'], input_directory=str(bundle['input_directory']))
    return dict(model=model, scores=scores, table=table, settings=settings,
                input_source=bundle['input_source'], input_directory=bundle['input_directory'],
                input_bundle=bundle['input_bundle'])


def save_ica(root, result, input_name):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='ica_',dir=root))
    np.save(folder/'scores.npy',result['scores'])
    result['table'].to_csv(folder/'image_scores.csv',index=False)
    model=result['model']
    np.savez_compressed(folder/'transform.npz',components=model.components_,mixing=model.mixing_,
                        mean=model.mean_,whitening=model.whitening_,
                        input_columns=result['settings']['input_columns'])
    settings=dict(result['settings'],representation=input_name,
                  transform='(selected_input - mean) @ components.T',
                  image_mapping='source and full image_name in image_scores.csv')
    (folder/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    (root/'latest_ica.json').write_text(json.dumps({'directory':str(folder.resolve())})+'\n')
    result['directory']=folder
    return folder


def load_ica(root, directory=None):
    """Restore numeric ICA arrays and the exact original image checkpoint."""
    from analysis_storage import resolve, load_input_bundle
    folder=resolve(root,directory,'latest_ica.json')
    settings=json.loads((folder/'settings.json').read_text())
    model=FastICA(n_components=settings['n_components'],whiten=settings['whiten'])
    with np.load(folder/'transform.npz',allow_pickle=False) as f:
        model.components_=f['components'].copy();model.mixing_=f['mixing'].copy()
        model.mean_=f['mean'].copy();model.whitening_=f['whitening'].copy()
    model.n_features_in_=model.components_.shape[1];model.n_iter_=settings['iterations']
    scores=np.load(folder/'scores.npy',allow_pickle=False)
    import pandas as pd
    table=pd.read_csv(folder/'image_scores.csv',dtype={'source':str,'image_name':str})
    if scores.shape!=(len(table),settings['n_components']):raise ValueError('ICA checkpoint dimensions disagree')
    return dict(model=model,scores=scores,table=table,settings=settings,directory=folder,
        input_source=settings['input_source'],input_directory=Path(settings['input_directory']),
        input_bundle=load_input_bundle(root,settings['input_source'],settings['input_directory']))
