# LatentSpace - Differentiating protein composition in cryo-EM images

Proteins can exist in different compositions which define the biological function. Here, we investigate a hydrogenase, which is either a dimer or a tetramer but only active in the dimer form. Conventional cryo-EM workflows struggle differentiating these in the 2D projections and commonly align them together. Thus, we try to separate them according to the information we have learnt in the ML4LS course.

The main analysis platform is the jupyter notebook `expeirment.ipynb`. Each cell of the notebook has a documentation of what is done and what data is needed. As input the user needs a particle stack in mrc format, a metadata file in relion star format and a pure noise dataset (from noise of the to investigated dataset, since noise is dependent on aquisition) in mrc format. 

```text
├── README.md
├── CONTRIBUTIONS.md
├── expeirment.ipynb
├── cwf_denoising/code.py
├── ja_notebook_validation.ipynb
│
├── latent_methods.py          # REQUIRED for 01; supply original file
├── lowpass_io.py              # REQUIRED for 01; supply original file
├── cwf_storage.py             # REQUIRED for 02b and 03; supply original file
├── particle_paired.py         # REQUIRED for 02b, 03 and 09; supply original file
├── particle_aggregation.py    # REQUIRED for 03; supply original file
│
├── streamlit/                 # Streamlit app folder, the currently run version relates to the forked repository LatentSpace_afi
└── outputs/                   # Generated datasets/models/figures; usually not tracked
```

## Setup

### 1. Install Python and Jupyter

The analysis was developed on **macOS with an Apple M3 and 8 GB RAM**. The notebooks use memory mapping and small batches for larger image arrays. A suitable recent Python environment with Jupyter is needed; exact package-version compatibility has **not been verified from a lockfile**.

Example installation:

```bash
cd /path/to/ml4ls
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab ipython numpy pandas matplotlib scipy \
    mrcfile starfile scikit-learn threadpoolctl joblib
jupyter lab
```

These are the main third-party imports used directly by the notebooks. **Check the dependencies of the original helper `.py` files too** (for example, a locally computed CWF workflow may have further requirements). This is a starter install command, **not** a tested or version-pinned `requirements.txt`.

### 2. Put source files in place

The notebooks expect, or inspect, the following filenames:

| Input | Notes |
|---|---|
| `particle_stack.mrc` | Original raw particle stack|
| `particle_stack.star` | STAR metadata with particle IDs and stack indexes |
| `noise_stack.mrc` | Pure noise stack |

**Critical ordering fact:** The CWF stack follows the **original raw MRC stack order**, **not the STAR row order**. Matching particles by `particle_id` and `stack_index_1based` is essential; never simply align STAR rows and CWF rows by their numerical position.

## Streamlit: Start the app

Run from the root of this repository in the same Python environment used for the notebook:

```bash
python -m pip install -r streamlit_requirements.txt
streamlit run streamlit_app.py
```

## How to use it

Run the notebook sections that save named clustering results into `LatentSpace_data`. Launch the app from this repository so it can find that folder. The app reads `clustering_index.json`, uses `analysis_storage.load_clustering`, and attempts to restore images using `cluster_examples.load_example_images`.

Information about the streamlit app: data output of the jupyter notebook was run on 4000 images (to keep the data small enough for smooth application running) which can be found in the `outputs` folder and are the basis for the streamlit app. Access the app here: https://latentspace.streamlit.app/



## Acknowledgements and scope

This README summarizes the main notebook `expeirment.ipynb`, the cwf denoising script `cwf_denoising/code.py` and validation `validation_folder`.
It also provides a streamlit app to investigate the results. The required helper modules are part of the original project implementation and need to be supplied separately.
Contributions of the different team members can be found in the CONTRIBUTIONS.md

## Citation

Cite this work as:

```
@software{LatentSpace, 
  author = {Xuelong Fan, Camille Penot, Khuanpiroon Ratanasopa, Anna-Lena Fischer},
  title = {LatentSpace - Differentiating protein composition in cryo-EM images},
  year = {2026},
  url = {https://github.com/bolodragon/LatentSpace}
}
```

# License 

This project is licensed under the MIT License. This means that anyone is free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright and license notice are included.

See the LICENSE file for the full license text.

Copyright (c) 2026 LatentSpace group

#### TO DO: software/package versions
#### TO DO: final tagged release / archived version
