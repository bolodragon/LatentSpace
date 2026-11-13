"""
Everything is loaded from the notebook's saved results (nothing is recomputed):
  outputs/clustering_index.json  ->  {"PCA kmeans": <folder>, "PCA DBSCAN": <folder>, "ICA kmeans": <folder>}
  <folder>/arrays.npz (selected_scores), table.json (image_name, cluster, ...), clusters.png, settings.json
"""

import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Cryo-EM particle explorer", page_icon="🔬", layout="wide")
APP_DIR = Path(__file__).resolve().parent


# =========================================================
# HEADER: movie 1 | title | movie 2
# =========================================================
@st.cache_data(show_spinner=False)
def make_video_html(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return f"<div style='text-align:center;padding:20px'>Video not found: {path.name}</div>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"""
    <video autoplay loop muted playsinline controls
           style="width:100%;height:220px;object-fit:contain;background:transparent">
        <source src="data:video/webm;base64,{b64}" type="video/webm">
    </video>"""


left_col, title_col, right_col = st.columns([1, 2, 1], vertical_alignment="center")
with left_col:
    components.html(make_video_html(str(APP_DIR / "movies" / "dimer_small.webm")), height=240)
with title_col:
    st.markdown(
        "<h1 style='text-align:center;font-size:2.3rem;line-height:1.2'>"
        "What protein composition is it?</h1>"
        "<p style='text-align:center;color:gray'>Differentiating protein composition in cryo-EM images </p>",
        unsafe_allow_html=True,
    )
with right_col:
    components.html(make_video_html(str(APP_DIR / "movies" / "tetramer_small.webm")), height=240)

st.markdown("##### Workflow\n\nImage denoising workflow starting from very experimental images without any possibility of differentiation between protein and noise to actually visualizing the particle")
workflow_path = APP_DIR / "images" / "workflow_ml4ls.png"
if workflow_path.exists():
    st.image(str(workflow_path), width="stretch"=True)
else:
    st.warning("Workflow image not found (images/workflow_ml4ls.png).")

st.divider()


# =========================================================
# LOADING SAVED RESULTS (same logic as the original script)
# =========================================================
def find_outputs_folder():
    for path in (APP_DIR, *APP_DIR.parents, Path.cwd(), *Path.cwd().parents):
        if (path / "outputs").is_dir():
            return path / "outputs"
    return None


@st.cache_data(show_spinner=False)
def read_index(outputs_folder: str) -> dict:
    return json.loads((Path(outputs_folder) / "clustering_index.json").read_text(encoding="utf-8"))


def resolve_checkpoint(outputs: Path, saved_path: str) -> Path:
    """Relative path, or an old absolute path from the computer the notebook ran on."""
    path = Path(saved_path)
    if path.exists():
        return path
    for candidate in (outputs / path.name, APP_DIR / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find checkpoint folder: {saved_path}")


@st.cache_data(show_spinner="Loading saved run…")
def load_run(outputs_folder: str, run_name: str):
    outputs = Path(outputs_folder)
    checkpoint = resolve_checkpoint(outputs, read_index(outputs_folder)[run_name])

    table = pd.read_json(checkpoint / "table.json", orient="table")
    with np.load(checkpoint / "arrays.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}

    if "selected_scores" not in arrays or arrays["selected_scores"].ndim != 2:
        raise ValueError(f"No 2-D 'selected_scores' in arrays.npz (found: {list(arrays)})")
    scores = arrays["selected_scores"]
    if len(scores) != len(table):
        raise ValueError("selected_scores and table have different numbers of rows.")

    plot = checkpoint / "clusters.png"
    return scores, table, (str(plot) if plot.exists() else None), variance_ratio_from(arrays)

def variance_ratio_from(arrays: dict):
    """Look for a 1-D explained-variance-ratio array."""
    for key, arr in arrays.items():
        k = key.lower().replace("_", "")
        if arr.ndim == 1 and ("explainedvarianceratio" in k or ("variance" in k and "ratio" in k)):
            return np.asarray(arr, dtype=float)
    return None


@st.cache_data(show_spinner=False)
def variance_ratio_from_model(folder: str):
    """Same lookup in model.npz of the PCA output folder."""
    model = Path(folder) / "model.npz"
    if not model.exists():
        return None
    with np.load(model, allow_pickle=False) as archive:
        return variance_ratio_from({k: archive[k] for k in archive.files})

def split_run_name(name: str):
    """'PCA kmeans' -> ('PCA', 'kmeans'); returns (None, name) if no method token."""
    match = re.search(r"(?<![a-z])(pca|ica)(?![a-z])", name, flags=re.IGNORECASE)
    if not match:
        return None, name
    method = match.group(1).upper()
    algo = (name[:match.start()] + name[match.end():]).strip(" _-") or name
    return method, algo


@st.cache_data(show_spinner=False)
def find_loadings(root: str) -> dict:
    """{folder: {start_index: png_path}} for every folder holding loadings_XXXX.png."""
    found = {}
    for f in Path(root).rglob("loadings_*.png"):
        m = re.search(r"loadings_(\d+)", f.name)
        if m:
            found.setdefault(str(f.parent), {})[int(m.group(1))] = str(f)
    return {k: dict(sorted(v.items())) for k, v in found.items()}


def loadings_for(method: str):
    """Loadings pages from the folder whose path mentions the method."""
    for folder, pages in find_loadings(str(APP_DIR)).items():
        if split_run_name(" ".join(Path(folder).parts[-3:]))[0] == method:
            return folder, pages
    return None, {}


def axis_label(i: int) -> str:
    """'PC1 (23.4%)' when the explained variance is known, otherwise 'PC1'."""
    if ratio is not None and i < len(ratio):
        return f"{comp_labels[i]} ({ratio[i]:.1%})"
    return comp_labels[i]


def scatter(df, x_label, y_label, title, color=None):
    fig = px.scatter(
        df, x="x", y="y", color=color,
        hover_name="particle",
        hover_data={"x": ":.3f", "y": ":.3f", "particle": False},
        labels={"x": x_label, "y": y_label},
        title=title,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch"=True)


def scatter_cluster(df, x_label, y_label, title, color=None, category_orders=None, color_map=None):
    fig = px.scatter(
        df, x="x", y="y", color=color,
        category_orders=category_orders,
        color_discrete_map=color_map,
        hover_name="particle",
        hover_data={"cluster": True, "x": ":.3f", "y": ":.3f", "particle": False},
        labels={"x": x_label, "y": y_label},
        title=title,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch"=True)


def frame(scores, table, xi, yi):
    if "image_name" in table.columns:
        # "path/to/stack.mrc@123" -> "123"  (names without "@" stay unchanged)
        particles = table["image_name"].astype(str).str.split("@", n=1).str[0].to_numpy()
    else:
        particles = np.arange(len(table)).astype(str)
    cluster = (table["cluster"].astype(str).to_numpy()
               if "cluster" in table.columns else np.full(len(table), "-"))
    return pd.DataFrame({"x": scores[:, xi], "y": scores[:, yi],
                         "particle": particles, "cluster": cluster})


outputs = find_outputs_folder()
if outputs is None or not (outputs / "clustering_index.json").exists():
    st.warning("Could not find outputs/clustering_index.json in the project.")
    st.stop()

entries = read_index(str(outputs))
if not entries:
    st.info("No saved clustering runs were found.")
    st.stop()

# method -> {algorithm label -> run name}
runs = {}
for run_name in sorted(entries):
    method, algo = split_run_name(run_name)
    runs.setdefault(method or "Latent space", {})[algo] = run_name
methods = sorted(runs)


# =========================================================
# SECTION 1: dimensionality reduction | scatter plot
# =========================================================
ctrl, plot = st.columns([1, 2.2])

with ctrl:
    st.markdown("**Dimensionality reduction method**")
    method = st.selectbox("Dimensionality reduction method", methods, label_visibility="collapsed")

# first saved run of this method supplies the embedding for the top plot
first_run = next(iter(runs[method].values()))
try:
    scores, table, _, ratio = load_run(str(outputs), first_run)
except Exception as e:
    st.error(f"Could not load saved run '{first_run}': {e}")
    st.stop()

loadings_folder, pages = loadings_for(method)
if ratio is None and loadings_folder:
    ratio = variance_ratio_from_model(loadings_folder)
if method != "PCA":
    ratio = None  # explained variance only makes sense for PCA

n_comp = scores.shape[1]
if n_comp < 2:
    st.error("Need at least two score components.")
    st.stop()

prefix = {"PCA": "PC", "ICA": "IC"}.get(method, "Component ")
comp_labels = [f"{prefix}{i + 1}" for i in range(n_comp)]

with ctrl:
    st.markdown(f"**{method} select component**")
    x_comp = st.selectbox("x", comp_labels, index=0)
    y_comp = st.selectbox("y", comp_labels, index=1)
    xi, yi = comp_labels.index(x_comp), comp_labels.index(y_comp)
    st.caption("Hover over a point to see its particle ID and x / y values.")

with plot:
    scatter(frame(scores, table, xi, yi), axis_label(xi), axis_label(yi),
            f"{method}: {x_comp} vs {y_comp}")
    if method == "PCA" and ratio is None:
        st.caption("Explained variance not found in arrays.npz or model.npz, so it is not shown on the axes.")


# =========================================================
# SECTION 2: important pixels (saved loadings images)
# =========================================================
with st.expander(f"{method} important pixels (click to show)"):
    if not pages:
        st.info(f"No loadings_XXXX.png files found for {method}.")
    else:
        starts = list(pages)
 
        def page_for(i: int) -> int:
            """Last loadings file whose start index is <= the component index."""
            return max([s for s in starts if s <= i], default=starts[0])
 
        # the dropdown opens on the page that contains the selected x component
        chosen = st.selectbox(
            "Loadings image",
            starts,
            index=starts.index(page_for(xi)),
            format_func=lambda s: Path(pages[s]).name,
        )
        st.image(pages[chosen], caption=Path(pages[chosen]).name, width="stretch"=True)
 
st.divider()


# =========================================================
# SECTION 3: clustering algorithm | clustered scatter plot
# =========================================================
ctrl2, plot2 = st.columns([1, 2.2])

with ctrl2:
    st.markdown("**Clustering algorithm**")
    algo = st.selectbox("Clustering algorithm", list(runs[method]), label_visibility="collapsed")

run_name = runs[method][algo]
try:
    scores2, table2, cluster_png, _ = load_run(str(outputs), run_name)
except Exception as e:
    st.error(f"Could not load saved run '{run_name}': {e}")
    st.stop()

n2 = scores2.shape[1]
xi2, yi2 = min(xi, n2 - 1), min(yi, n2 - 1)
if (xi2, yi2) != (xi, yi):
    st.caption(f"This run only has {n2} score components; axes were clamped.")

with ctrl2:
    if "cluster" in table2.columns:
        counts = table2["cluster"].value_counts().sort_index().rename("particles")
        if "dbscan" in algo.lower():
            counts = counts[counts.index.astype(str).isin(["1", "-1"])]
        st.caption(f"{len(table2):,} particles · cluster sizes (-1 = noise)")
        st.dataframe(counts, width="stretch"=True)

with plot2:
    df2 = frame(scores2, table2, xi2, yi2)
    cmap = None
 
    def numeric_first(label: str):
        """Sort key: numeric labels in numeric order."""
        try:
            return (0, int(label), "")
        except ValueError:
            return (1, 0, label)
 
    # legend order: -1, 0, 1, 2, ... (plain string sorting would give "10" before "2")
    order = {"cluster": sorted(df2["cluster"].unique(), key=numeric_first)}
 
    if "dbscan" in algo.lower():
        # DBSCAN: noise is drawn first = in the background
        df2 = df2[df2["cluster"].isin(["1", "-1"])]
        order = {"cluster": ["-1", "1"]}
        cmap = {"-1": "lightgray", "1": "#1f77b4"}
        st.caption(f"Showing only cluster 1 and noise (-1): {len(df2):,} of {len(table2):,} particles.")
 
    scatter_cluster(df2, axis_label(xi2), axis_label(yi2),
            f"{run_name}: {comp_labels[xi2]} vs {comp_labels[yi2]}",
            color="cluster", category_orders=order, color_map=cmap)
 

if cluster_png:
    with st.expander("Clustering plot saved by the notebook"):
        st.image(cluster_png, width="stretch"=True)

st.download_button(
    "Download saved assignments (CSV)",
    table2.to_csv(index=False).encode(),
    f"{run_name}_assignments.csv",
    "text/csv",
)
