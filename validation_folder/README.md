# Cryo-EM Particle Preprocessing and Unsupervised Validation

This folder contains four Jupyter notebooks documenting an exploratory single-particle cryo-electron microscopy (cryo-EM) workflow. The analysis uses **57,107 particles**, prepares features from Camille's CWF-processed images, and evaluates PCA and K-means clustering.

## Notebooks

Run or review the notebooks in this order:

1. **`01_raw_preprocessing_2_.ipynb` — Raw preprocessing and frozen data split.** Inspects raw MRC images and STAR metadata, explores recentering and Gaussian low-pass filtering, and records a fixed particle-level split: **5,711 held-out test particles (Fold 0)** and **three development folds of 17,132 particles each (Folds 1–3)**.
2. **`02b_import_camille_cwf_2_.ipynb` — CWF import and particle mapping.** Documents the externally processed CWF stack, maps particle identities to original MRC stack positions, and produces full cropped original and derivative image checkpoints. Separate optional subset-import/normalization sections are not the final full-data route.
3. **`03_crop_align_features_2_.ipynb` — Image alignment and feature extraction.** Prepares paired original and derivative images, aligns particles, extracts radial-sector and FFT descriptors, and saves combined features for downstream analysis.
4. **`09_data_validation_2_.ipynb` — Validation and clustering.** Checks feature-to-particle correspondence, performs three-fold PCA/K-means validation with training-only preprocessing, examines brightness-related effects, evaluates a 48-feature brightness-control representation, and applies a frozen final model to the held-out test set. The later post-test parameter search is exploratory.

## Critical particle-ordering rule

**The CWF stack follows the original raw MRC stack order, not the STAR metadata row order.** Particle matching must use `particle_id` and `stack_index_1based` (see `outputs/validation/cwf_particle_mapping.csv`). **Never match STAR and CWF rows by their numerical positions.** The CWF ordering relies on confirmation from the data producer; the notebook checks the resulting mapping for internal consistency.

## Data and reproducibility

The notebooks require the original input files, locally available Python helper modules, and generated checkpoints under `outputs/`. Large MRC stacks and NumPy arrays are not included in this notebook folder. The frozen split is recorded in `outputs/validation/cv_3fold_with_unknown_test.csv` and should be reused rather than regenerated when reproducing the reported results.

**Interpretation:** The identified clusters describe variation in 2D particle images; they are not, by themselves, evidence of distinct biological conformations. Validation is particle-level, not micrograph-level.
