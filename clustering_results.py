"""Named clustering results independent of image display and PCA variables."""
import numpy as np


def register_clustering_result(registry, name, table, *, input_source,
                               input_directory, input_bundle=None, parameters=None):
    """Store a snapshot; rows carry full image IDs and integer cluster labels.

    Labels may be nonconsecutive or negative (e.g. DBSCAN noise label -1).
    """
    required = ['source', 'image_name', 'source_row_1based', 'cluster']
    missing = set(required)-set(table.columns)
    if missing: raise ValueError(f'Missing clustering columns: {sorted(missing)}')
    result = table.copy(deep=True)
    if not len(result) or result[required].isna().any().any():
        raise ValueError('Clustering rows must be nonempty and have complete IDs and labels')
    labels = result['cluster'].to_numpy()
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Cluster labels must be integers; nonconsecutive and negative labels are allowed')
    if not result['source'].isin(['particle','noise']).all():
        raise ValueError('Sources must be particle or noise')
    if result.duplicated(['source','image_name']).any():
        raise ValueError('Each source/image ID must occur only once')
    if not name or not isinstance(name,str): raise ValueError('Use a nonempty method name')
    registry[name] = dict(table=result,input_source=input_source,input_directory=input_directory,
                          input_bundle=input_bundle,parameters=dict(parameters or {}))
    return registry[name]
