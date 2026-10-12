"""Read an MRC image or volume.

Install dependencies: python -m pip install mrcfile numpy matplotlib
Usage: python LatentSpace/experiment.py [path/to/file.mrc]
"""

import argparse
from pathlib import Path

import mrcfile
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_MRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "LatentSpace_data"
    / "cryosparc_P543_templates_selected.mrc"
)


def read_mrc(file_path: str | Path) -> np.ndarray:
    """Load MRC data into a NumPy array, preserving its stored axis order."""
    with mrcfile.open(str(file_path), mode="r") as mrc:
        # Copy so the array remains usable after the file is closed.
        return mrc.data.copy()


def display_mrc(data: np.ndarray) -> None:
    """Display a 2D image or a gallery of slices from a 3D MRC array."""
    if data.ndim == 2:
        images = data[np.newaxis, ...]
    elif data.ndim == 3:
        images = data
    else:
        raise ValueError(f"Expected 2D or 3D MRC data, got shape {data.shape}")

    count = len(images)
    columns = min(5, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False
    )
    for index, ax in enumerate(axes.flat):
        if index < count:
            ax.imshow(images[index], cmap="gray", origin="upper")
            ax.set_title(f"Image / slice {index + 1}")
        ax.axis("off")
    fig.suptitle("MRC images / slices")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read an MRC image or volume.")
    parser.add_argument(
        "file_path", type=Path, nargs="?", default=DEFAULT_MRC_PATH,
        help="Path to the MRC file (defaults to the selected cryoSPARC templates)",
    )
    args = parser.parse_args()

    data = read_mrc(args.file_path)
    print(f"Loaded: {args.file_path}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    display_mrc(data)


if __name__ == "__main__":
    main()
