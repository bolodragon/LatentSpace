# v1.0 - 20261113

Initial release of LatentSpace, an exploratory workflow for investigating protein composition in cryo-EM particle images.

- Notebook workflow (`experiment.ipyn`) for preprocessing, CWF denoising, feature extraction, PCA, ICA, and clustering.
- Validation notebooks (`validation_folder`) for particle mapping, feature preparation, and cross-validation.
- Saved analysis checkpoints and a Streamlit app for exploring clusters and particle images.
- Python dependencies collected in `requirements.txt` for the notebooks, helper modules, and app.

Running the analysis requires external particle and noise stacks in MRC format and particle metadata in STAR format.
