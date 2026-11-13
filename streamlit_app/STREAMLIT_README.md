# Cryo-EM Streamlit explorer

## Start the app

Run from the root of this repository in the same Python environment used for the notebook:

```bash
python -m pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py
```

## Two ways to use it

### 1. Feature CSV (flexible mode)

Upload a CSV with one row per particle:
- an ID column such as `particle_id` or `image_name`;
- numeric feature columns, ideally the same descriptors used by your notebook;
- optionally a `cluster` column to inspect precomputed labels;
- optionally `known_class` for reference labels (kept out of feature calculations).

An optional ZIP can contain particle images. Filenames should match the particle IDs (or their file stems). The app provides PCA, ICA and UMAP views; K-means, DBSCAN and HDBSCAN; cluster inspection; representative particles; assignment export; and existing-particle lookup.

### 2. Saved notebook clustering results

Run the notebook sections that save named clustering results into `LatentSpace_data`. Launch the app from this repository so it can find that folder. The app reads `clustering_index.json`, uses `analysis_storage.load_clustering`, and attempts to restore images using `cluster_examples.load_example_images`.

## New-image assignment

Existing particle lookup is exact because the saved table contains its cluster label. For an image that is not already in the dataset, create `app_feature_pipeline.py` in this repository with a function:

```python
def extract_features(image):
    # image is a PIL grayscale image.
    # Apply the SAME preprocessing and feature extraction used to create
    # the feature CSV, and return a 1D numeric vector in the same column order.
    return feature_vector
```

The app then uses standardized nearest-neighbour distance to find the closest reference particle and displays its cluster representative. The displayed distance is not a calibrated confidence or a biological dimer/tetramer probability. Do not use a different preprocessing pipeline or feature order.

## Scientific interpretation

PCA, ICA, UMAP and density/centroid clustering reveal structure in the chosen representation. Cluster numbers are arbitrary, DBSCAN/HDBSCAN noise is labelled -1, and unsupervised clusters do not automatically mean dimer or tetramer. Biological interpretation needs independently validated labels or structural evidence.
