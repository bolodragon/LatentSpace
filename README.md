# LatentSpace - Differentiating protein composition in cryo-EM images

Proteins can exist in different compositions which define the biological function. Here, we investigate a hydrogenase, which is either a dimer or a tetramer but only active in the dimer form. Conventional cryo-EM workflows struggle differentiating these in the 2D projections and commonly align them together. Thus, we try to separate them according to the information we have learnt in the ML4LS course.

The main analysis platform is the jupyter notebook "expeirment.ipynb". Each cell of the notebook has a documentation of what is done and what data is needed. As input the user needs a particle stack in mrc format, a metadata file in relion star format and a pure noise dataset (from noise of the to investigated dataset, since noise is dependent on aquisition) in mrc format. 

```text
├── README.md
├── CONTRIBUTIONS.md
├── expeirment.ipynb
├── camille_code_cwf.ipynb/py
├── ja_notebook_validation.ipynb
│
├── latent_methods.py          # REQUIRED for 01; supply original file
├── lowpass_io.py              # REQUIRED for 01; supply original file
├── cwf_storage.py             # REQUIRED for 02b and 03; supply original file
├── particle_paired.py         # REQUIRED for 02b, 03 and 09; supply original file
├── particle_aggregation.py    # REQUIRED for 03; supply original file
│
├── streamlit_app.py           # Streamlit app file
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

| Input | Used in | Notes |
|---|---|---|
| `particle_stack.mrc` | 01, 02b | Original raw particle stack|
| `particle_stack.star` | 01 | STAR metadata with particle IDs and stack indexes |
| `noise_stack.mrc` | 01 | Pure noise stack |

**Critical ordering fact:** The CWF stack follows the **original raw MRC stack order**, **not the STAR row order**. Matching particles by `particle_id` and `stack_index_1based` is essential; never simply align STAR rows and CWF rows by their numerical position.


XXXXX - change in notebook - XXXXX
Some older exploratory cells refer to data underneath `LatentSpace_data/` or `outputs/cwf_from_camille/`. Check those cells against your local layout; the main validated full-stack path used by this project is `particles_cwf.mrc` at the project root.
XXXXX - change in notebook - XXXXX
### 3. Update machine-specific paths
XXXXX - change in notebook - XXXXX
Several uploaded notebooks contain hard-coded paths such as:
XXXXX - change in notebook - XXXXX
```python
PROJECT_ROOT = Path("/Users/kr/ml4ls")
```
XXXXX - change in notebook - XXXXX
**Change these paths to your own repository location** before running. In particular, inspect:
XXXXX - change in notebook - XXXXX
- `project_root`, `PROJECT_ROOT`, `seg_root`, and any module search paths;
- `CHECKPOINT_DIRECTORY` in Notebook 09 (the recorded run uses `outputs/paired_1yn3a_7i`);
- `final_model_dir_08` in Notebook 08 (the recorded run uses `outputs/validation_09/20261109T175733Z/...`);
- input stack paths and other paths in legacy/diagnostic cells.
XXXXX - change in notebook - XXXXX
The checkpoint directory names above identify **existing example runs**, not universal filenames automatically created on every computer.


#### TO DO: merge LatentSpace_afi for streamlit files!!! 
#### Information about the streamlit app: data output of the jupyter notebook run on 4000 images can be found in the outputs folder and are the basis for the streamlit app. Access the app here: https://latentspace.streamlit.app/



## Acknowledgements and scope

This README summarizes the main notebook `expeirment.ipynb`, the cwf denoising script `xxx_camille.py` and validation `validation_folder`. The required helper modules are part of the original project implementation and need to be supplied separately.
Contributions of the different team members can be found in the CONTRIBUTIONS.md




#### TO DO: software/package versions
#### TO DO: licence and citation information
#### TO DO: final tagged release / archived version
