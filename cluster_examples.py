"""Resolve cluster examples from selectable datasets using full image IDs."""
from pathlib import Path
import numpy as np
from cwf_storage import load_cluster_cwf_images, select_cwf_directory
from particle_segmentation import load_section3_results
from particle_alignment import load_aligned_results
from particle_aggregation import load_aggregated_results


def load_example_images(root, table, dataset='source', directory=None,
                        fitted_source=None, fitted_directory=None, fitted_bundle=None,
                        angle=None):
    root = Path(root)
    if dataset in ('source', 'own', 'camille', 'normalized_camille'):
        if dataset == 'source':
            folder = (Path(fitted_bundle['settings']['source_checkpoint'])
                      if fitted_source in ('aggregated','aligned','rotated','rotated_average') else Path(fitted_directory))
            if directory is not None:
                folder = Path(directory)
                if not folder.is_absolute(): folder = root/folder
        else:
            folder = select_cwf_directory(root, dataset, directory)
        return load_cluster_cwf_images(folder, table), folder, f'{dataset} CWF images'
    if dataset not in ('aggregated', 'aligned', 'rotated', 'rotated_average', 'masked'):
        raise ValueError('Choose source, own, camille, normalized_camille, aggregated, aligned, rotated, rotated_average, or masked')
    if not (table['source'] == 'particle').all():
        raise ValueError('This dataset has no noise controls; select source or use particle-only clustering')
    if directory is None:
        if dataset == 'aggregated' and fitted_source == 'aggregated': directory = fitted_directory
        if dataset == 'aligned' and fitted_source == 'aggregated': directory = fitted_bundle['settings']['aligned_directory']
        if dataset == 'aligned' and fitted_source == 'aligned': directory = fitted_directory
        elif dataset in ('rotated','rotated_average','masked') and fitted_source in ('rotated','rotated_average'):
            directory = fitted_directory
    bundle = (load_aggregated_results(root, directory) if dataset == 'aggregated' else
              load_aligned_results(root, directory) if dataset == 'aligned'
              else load_section3_results(root, directory))
    if dataset == 'aggregated': images = bundle['vectors'][:, bundle['sector_map']]
    elif dataset == 'aligned': images = bundle['images']
    elif dataset == 'rotated_average': images = bundle['averaged_images']
    elif dataset == 'masked': images = bundle['masked_images']
    else:
        angles = np.asarray(bundle['angles_degrees'])
        selected = angles[0] if angle is None else angle
        matches = np.flatnonzero(np.isclose(angles % 360, selected % 360))
        if not len(matches): raise ValueError(f'Choose an angle from {angles.tolist()}')
        images = bundle['orientation_images'][:, matches[0]]
        dataset = f'rotated at {angles[matches[0]]:g}°'
    if images is None:
        raise ValueError('This older checkpoint does not contain the requested images; rerun its save step')
    names = bundle['image_names'].astype(str).tolist()
    if len(set(names)) != len(names): raise ValueError('Duplicate dataset image IDs')
    lookup = {name:i for i,name in enumerate(names)}
    try: rows = [lookup[str(name)] for name in table['image_name']]
    except KeyError as error:
        raise ValueError(f'Image ID {error.args[0]} is missing from the chosen dataset; select a matching checkpoint') from error
    return np.asarray(images[rows]), bundle['directory'], dataset
