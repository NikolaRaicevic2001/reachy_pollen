"""
Load a LeRobot Parquet chunk, print a short summary, and plot angle-like
vectors (observation.state, action) vs time with joint names from meta/info.json.
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Hyperparameters
DEFAULT_DATA_DIR = os.environ.get("LEROBOT_DATA_CHUNK", r"C:\Users\nikra\.cache\huggingface\lerobot\pollen_robotics\record_test\data\chunk-000")

################################
####### Helper Functions #######
################################
def load_parquet_dir(data_dir: Path) -> pd.DataFrame:
    """ Load all .parquet files in the given directory and concatenate them into a single DataFrame """
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if "index" in df.columns:
        df = df.sort_values("index", kind="mergesort").reset_index(drop=True)
    return df


def resolve_dataset_root(data_dir: Path) -> Path | None:
    """ LeRobot layout: <root>/meta/info.json and <root>/data/chunk-*/ """
    for candidate in (data_dir.parent.parent, data_dir.parent, data_dir):
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    return None


def load_feature_names(dataset_root: Path, column: str) -> list[str] | None:
    """ Load feature names from info.json """
    path = dataset_root / "meta" / "info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    feat = info.get("features", {}).get(column)
    if not isinstance(feat, dict):
        return None
    names = feat.get("names")
    if not isinstance(names, list) or not names:
        return None
    return [str(n) for n in names]


def joint_labels_for_column(dataset_root: Path | None, column: str, n_dims: int) -> list[str]:
    """ Get joint labels for a given column """
    default = [f"dim_{i}" for i in range(n_dims)]
    if dataset_root is None:
        return default
    loaded = load_feature_names(dataset_root, column)
    if not loaded:
        return default
    if len(loaded) != n_dims:
        print(
            f"Warning: {column} has {n_dims} dims but info.json lists {len(loaded)} names; "
            "using names where they align, else dim_*."
        )
    out = list(default)
    for i in range(min(n_dims, len(loaded))):
        out[i] = loaded[i]
    return out


def parse_int_list(s: str | None) -> list[int] | None:
    """ Parse a comma-separated list of integers """
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_name_filters(s: str | None) -> list[str] | None:
    """ Parse a comma-separated list of substrings """
    if s is None or not str(s).strip():
        return None
    return [p.strip() for p in str(s).split(",") if p.strip()]


def resolve_plot_indices(n_dims: int, labels: list[str], indices: list[int] | None, name_filters: list[str] | None) -> tuple[list[int], list[str]]:
    """ Resolve indices and labels for plotting """
    if indices is not None:
        bad = [i for i in indices if i < 0 or i >= n_dims]
        if bad:
            print(f"Warning: ignoring out-of-range indices (valid 0..{n_dims - 1}): {bad}")
        chosen = sorted({i for i in indices if 0 <= i < n_dims})
        if not chosen:
            raise SystemExit("No valid --indices in range for this column.")
        return chosen, [labels[i] for i in chosen]

    if name_filters is not None:
        chosen = []
        for i in range(n_dims):
            name_l = labels[i].lower()
            if any(f.lower() in name_l or name_l == f.lower() for f in name_filters):
                chosen.append(i)
        if not chosen:
            raise SystemExit(
                f"No joints matched --joint-names {name_filters!r}. "
                "Use --list-joints to see names."
            )
        return chosen, [labels[i] for i in chosen]

    return list(range(n_dims)), labels


def print_joint_list(dataset_root: Path | None, df: pd.DataFrame) -> None:
    """ Print index and joint name per dimension """
    print("\n--- Joint names (from meta/info.json if available) ---")
    for col in ("observation.state", "action"):
        if col not in df.columns:
            continue
        arr = series_to_2d(df[col])
        if arr is None:
            print(f"{col}: (could not read vector column)")
            continue
        labels = joint_labels_for_column(dataset_root, col, arr.shape[1])
        print(f"\n{col} ({arr.shape[1]} dims):")
        for i, name in enumerate(labels):
            print(f"  [{i:3d}] {name}")


def print_summary(df: pd.DataFrame) -> None:
    """ Print summary of the DataFrame """
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Columns ---")
    print(df.dtypes)
    print("\n--- info() ---")
    df.info()
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        print("\n--- describe() (numeric columns) ---")
        print(num.describe().T.to_string())
    print("\n--- head(3) ---")
    print(df.head(3))


def series_to_2d(series: pd.Series) -> np.ndarray | None:
    """ Convert a pandas Series to a 2D numpy array """
    if series.dtype == object or series.dtype == "O":
        values = series.tolist()
    else:
        values = series.to_numpy()
    if not values:
        return None
    v0 = values[0]
    if isinstance(v0, np.ndarray):
        out = np.stack([np.asarray(x, dtype=np.float64) for x in values])
        return out if out.ndim == 2 else None
    if isinstance(v0, (list, tuple)):
        try:
            return np.stack([np.asarray(x, dtype=np.float64) for x in values])
        except (ValueError, TypeError):
            return None
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    return arr.reshape(-1, 1)


def time_values(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """ Get time values from the DataFrame """
    if "timestamp" in df.columns:
        t = np.asarray(df["timestamp"], dtype=np.float64)
        if np.all(np.isfinite(t)):
            return t - t[0], "time (s) from start"
    if "frame_index" in df.columns:
        return np.asarray(df["frame_index"], dtype=np.float64), "frame_index"
    return np.arange(len(df), dtype=np.float64), "sample index"


def _legend_ncol(n: int) -> int:
    """ Get number of columns for the legend """
    return 1 if n <= 8 else 2 if n <= 16 else 3


def plot_angles_over_time(t: np.ndarray, arr: np.ndarray, title: str, ax: plt.Axes, dim_indices: list[int], legend_labels: list[str]) -> None:
    """ Plot angles over time """
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(dim_indices), 1)))
    for k, j in enumerate(dim_indices):
        ax.plot(
            t,
            arr[:, j],
            linewidth=1.2,
            alpha=0.85,
            color=cmap[k % len(cmap)],
            label=legend_labels[k],
        )
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=_legend_ncol(len(dim_indices)), framealpha=0.9)

################################
######### Main Function ########
################################
def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LeRobot Parquet chunk and plot joint trajectories vs time.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR, help="Folder with file-*.parquet (usually data/chunk-000)")
    parser.add_argument("--no-plot", action="store_true", help="Only print summary, do not open figures")
    parser.add_argument("--list-joints", action="store_true", help="Print index and joint name per dimension, then exit")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated 0-based dimension indices to plot, e.g. 0,3,7 (overrides --joint-names)")
    parser.add_argument("--joint-names", type=str, default=None, help="Comma-separated substrings; plot joints whose label contains any substring (case-insensitive), e.g. l_shoulder,r_wrist")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    df = load_parquet_dir(data_dir)
    dataset_root = resolve_dataset_root(data_dir)
    if dataset_root is None:
        print("\nNote: meta/info.json not found next to data/; joint names will be dim_0, dim_1, ...")
    else:
        print(f"\nDataset root (for joint names): {dataset_root}")

    print_summary(df)

    if args.list_joints:
        print_joint_list(dataset_root, df)
        return

    indices_arg = parse_int_list(args.indices)
    name_filters = parse_name_filters(args.joint_names)
    if indices_arg is not None and name_filters is not None:
        print("Note: both --indices and --joint-names set; using --indices only.")

    if args.no_plot:
        return

    t, xlabel = time_values(df)
    cols = [c for c in ("observation.state", "action") if c in df.columns]
    if not cols:
        print("\nNo observation.state or action column to plot.")
        return

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.8 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])
    for ax, col in zip(axes, cols, strict=True):
        arr = series_to_2d(df[col])
        if arr is None:
            ax.set_title(f"{col} (could not convert to 2D floats)")
            continue
        labels = joint_labels_for_column(dataset_root, col, arr.shape[1])
        try:
            dim_idx, leg = resolve_plot_indices(arr.shape[1], labels, indices_arg, name_filters if indices_arg is None else None)
        except SystemExit as e:
            raise SystemExit(f"{col}: {e.args[0]}") from e
        title = f"{col} ({len(dim_idx)} of {arr.shape[1]} joints)"
        plot_angles_over_time(t, arr, title, ax, dim_idx, leg)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
