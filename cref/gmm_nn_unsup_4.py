"""
Unsupervised GMM clustering for ultra-faint Darkfield CR-39 track scans.
Implements spatial welding (Morphological Closing) to reconstruct
fragmented track signals before area filtering, optional post-GMM watershed
splitting, saliency-based artifact tagging, and count comparison vs ImageJ.
"""

import argparse
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from scipy import ndimage as ndi
from skimage import segmentation
from skimage.feature import peak_local_max
from skimage.morphology import h_minima
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# =====================================================================
# CREF filename parsing and image discovery
# =====================================================================
CREF_FILENAME_RE = re.compile(
    r"^CREF_(?P<lab>[^_]+)_(?P<period>T[123])_(?P<position>[ABC]\d+)_(?P<base>LBS\d+)\.jpg$"
)


@dataclass
class CrefMeta:
    lab: str
    period: str
    position: str
    base: str

    @property
    def base_filename(self) -> str:
        return f"{self.base}.jpg"


def parse_cref_filename(name: str) -> CrefMeta | None:
    match = CREF_FILENAME_RE.match(name)
    if not match:
        return None
    return CrefMeta(
        lab=match.group("lab"),
        period=match.group("period"),
        position=match.group("position"),
        base=match.group("base"),
    )


def period_to_code(period: int) -> str:
    if period not in (1, 2, 3):
        raise ValueError(f"filter_period must be 1, 2, or 3; got {period}")
    return f"T{period}"


def image_display_name(filename: str) -> str:
    meta = parse_cref_filename(filename)
    if meta:
        return f"{meta.lab}_{meta.period}_{meta.position}"
    return Path(filename).stem


def _matches_cref_filters(meta: CrefMeta, cfg: "Config") -> bool:
    if cfg.filter_lab is not None and meta.lab != cfg.filter_lab:
        return False
    if cfg.filter_period is not None and meta.period != period_to_code(cfg.filter_period):
        return False
    if cfg.filter_position_group is not None:
        group = cfg.filter_position_group.upper()
        if not meta.position.upper().startswith(group):
            return False
    return True


def discover_images(cfg: "Config") -> list[Path]:
    paths = sorted(cfg.data_dir.glob(cfg.image_glob))
    if cfg.filter_lab is None and cfg.filter_period is None and cfg.filter_position_group is None:
        return paths
    filtered = []
    for path in paths:
        meta = parse_cref_filename(path.name)
        if meta is not None and _matches_cref_filters(meta, cfg):
            filtered.append(path)
    return filtered


def summarize_cref_inventory(data_dir: Path) -> dict:
    labs, periods, position_groups = set(), set(), set()
    total = 0
    for path in sorted(data_dir.glob("CREF_*.jpg")):
        meta = parse_cref_filename(path.name)
        if meta is None:
            continue
        total += 1
        labs.add(meta.lab)
        periods.add(meta.period)
        position_groups.add(meta.position[0])
    return {
        "labs": sorted(labs),
        "periods": sorted(periods),
        "position_groups": sorted(position_groups),
        "total_files": total,
    }


def lookup_macro_count(macro_map: dict, image: str):
    if image in macro_map:
        return macro_map[image]
    meta = parse_cref_filename(image)
    if meta:
        for key in (meta.base_filename, meta.base):
            if key in macro_map:
                return macro_map[key]
    return np.nan


def resolve_visual_roi_image(cfg: "Config", image_names: list[str]) -> str | None:
    if cfg.visual_roi_image and cfg.visual_roi_image in image_names:
        return cfg.visual_roi_image
    return image_names[0] if image_names else None


# =====================================================================
# CREF per-image enrichment and cross-group summaries
# =====================================================================
GMM_COUNT_COLS = ["imagej_count", "gmm_count", "gmm_ws_count", "gmm_ws_clean_count"]
RADON_COUNT_COLS = ["imagej_count", "macro_repro", "improved", "nn_clean_count"]

CREF_SUMMARY_SPECS: list[tuple[str, list[str]]] = [
    ("grand_total", []),
    ("by_period", ["period"]),
    ("by_lab_period", ["lab", "period"]),
    ("by_lab", ["lab"]),
    ("by_position_group", ["position_group"]),
    ("by_lab_period_position", ["lab", "period", "position_group"]),
    ("by_lab_position", ["lab", "position_group"]),
    ("by_period_position", ["period", "position_group"]),
]


def enrich_with_cref_metadata(df: pd.DataFrame, image_col: str = "image") -> pd.DataFrame:
    """Add lab, period, position, position_group, base parsed from CREF filenames."""
    out = df.copy()
    records = []
    for name in out[image_col].astype(str):
        meta = parse_cref_filename(name)
        if meta is None:
            records.append({
                "lab": None,
                "period": None,
                "period_num": None,
                "position": None,
                "position_group": None,
                "base": None,
            })
        else:
            records.append({
                "lab": meta.lab,
                "period": meta.period,
                "period_num": int(meta.period[1]),
                "position": meta.position,
                "position_group": meta.position[0],
                "base": meta.base,
            })
    meta_df = pd.DataFrame(records)
    for col in meta_df.columns:
        out[col] = meta_df[col].values

    meta_cols = ["lab", "period", "period_num", "position", "position_group", "base"]
    other_cols = [c for c in out.columns if c not in meta_cols]
    if image_col in other_cols:
        idx = other_cols.index(image_col) + 1
        ordered = other_cols[:idx] + meta_cols + [c for c in other_cols[idx:] if c not in meta_cols]
    else:
        ordered = other_cols + meta_cols
    return out[ordered]


def aggregate_cref_counts(
    df: pd.DataFrame,
    group_cols: list[str],
    count_cols: list[str],
) -> pd.DataFrame:
    """Aggregate count columns by group; returns n_images, sum_*, mean_* per group."""
    count_cols = [c for c in count_cols if c in df.columns]
    numeric = df[count_cols].apply(pd.to_numeric, errors="coerce")

    if not group_cols:
        row: dict = {"n_images": len(df)}
        for col in count_cols:
            row[f"sum_{col}"] = numeric[col].sum()
            row[f"mean_{col}"] = numeric[col].mean()
        return pd.DataFrame([row])

    work = df[group_cols].copy()
    for col in count_cols:
        work[col] = numeric[col].values
    grouped = work.groupby(group_cols, dropna=False)
    parts = [grouped.size().rename("n_images")]
    for col in count_cols:
        parts.append(grouped[col].sum().rename(f"sum_{col}"))
        parts.append(grouped[col].mean().rename(f"mean_{col}"))
    return pd.concat(parts, axis=1).reset_index()


def build_cref_summaries(df: pd.DataFrame, count_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Build all standard CREF grouping summary tables."""
    return {
        key: aggregate_cref_counts(df, group_cols, count_cols)
        for key, group_cols in CREF_SUMMARY_SPECS
    }


def save_cref_summaries(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str = "",
) -> Path:
    """Write summary tables to output_dir/summaries/{prefix}_*.csv."""
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for key, table in summaries.items():
        name = f"{prefix}_{key}.csv" if prefix else f"{key}.csv"
        table.to_csv(summary_dir / name, index=False)
    return summary_dir


def pivot_lab_period(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot lab x period matrix (sums) for heatmaps."""
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not in dataframe")
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    return work.pivot_table(
        index="lab",
        columns="period",
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )


POSITION_GROUPS = ("A", "B", "C")
POSITION_SLOTS = (1, 2, 3, 4, 5)


def position_slot_matrix(
    df: pd.DataFrame,
    lab: str,
    period: str,
    value_col: str,
) -> pd.DataFrame:
    """3×5 matrix (position_group × slot) for one lab-period tile."""
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not in dataframe")
    sub = df[(df["lab"] == lab) & (df["period"] == period)].copy()
    if sub.empty:
        return pd.DataFrame(
            np.nan,
            index=list(POSITION_GROUPS),
            columns=list(POSITION_SLOTS),
        )
    if "position_slot" not in sub.columns:
        sub["position_slot"] = sub["position"].astype(str).str[1:].astype(int)
    if "position_group" not in sub.columns:
        sub["position_group"] = sub["position"].astype(str).str[0]
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    pivot = sub.pivot_table(
        index="position_group",
        columns="position_slot",
        values=value_col,
        aggfunc="sum",
    )
    return pivot.reindex(index=list(POSITION_GROUPS), columns=list(POSITION_SLOTS))


def plot_lab_period_position_heatmaps(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    save_path: Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    tile_stats: pd.DataFrame | None = None,
    tile_stats_col: str = "std",
    inner_grid: bool = False,
):
    """Mosaic: lab × period tiles, each tile a 3×5 position heatmap."""
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    labs = sorted(work["lab"].dropna().unique())
    periods = sorted(work["period"].dropna().unique())
    if vmin is None:
        vmin = float(work[value_col].min())
    if vmax is None:
        vmax = float(work[value_col].max())

    stats_lookup: dict[tuple[str, str], float] = {}
    if tile_stats is not None and {"lab", "period", tile_stats_col}.issubset(tile_stats.columns):
        for row in tile_stats.itertuples(index=False):
            stats_lookup[(str(row.lab), str(row.period))] = float(getattr(row, tile_stats_col))

    n_labs, n_periods = len(labs), len(periods)
    fig, axes = plt.subplots(
        n_labs,
        n_periods,
        figsize=(2.8 * n_periods, 2.6 * n_labs),
        squeeze=False,
        constrained_layout=True,
    )
    try:
        plot_cmap = plt.colormaps[cmap].copy()
    except KeyError:
        import seaborn as sns

        plot_cmap = sns.color_palette(cmap, as_cmap=True)
    plot_cmap.set_bad(color="#e0e0e0")

    last_im = None
    for i, lab in enumerate(labs):
        for j, period in enumerate(periods):
            ax = axes[i, j]
            mat = position_slot_matrix(work, lab, period, value_col)
            last_im = ax.imshow(
                mat.values,
                aspect="auto",
                cmap=plot_cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if inner_grid:
                ax.set_xticks(np.arange(-0.5, len(POSITION_SLOTS), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(POSITION_GROUPS), 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=0.9, alpha=0.65)
                ax.tick_params(which="minor", bottom=False, left=False)
            tile_title = f"{lab} {period}"
            sigma = stats_lookup.get((lab, period))
            if sigma is not None and np.isfinite(sigma):
                tile_title += f"\nσ={sigma:.1f}"
            ax.set_title(tile_title, fontsize=8)
            if j == 0:
                ax.set_yticks(range(len(POSITION_GROUPS)), labels=POSITION_GROUPS)
            else:
                ax.set_yticks([])
            if i == n_labs - 1:
                ax.set_xticks(range(len(POSITION_SLOTS)), labels=POSITION_SLOTS)
                ax.set_xlabel("slot")
            else:
                ax.set_xticks([])

    fig.suptitle(f"{title} — position detail (A/B/C × 1–5)", fontsize=12, y=1.02)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def add_within_tile_centered(
    df: pd.DataFrame,
    value_col: str,
    mode: str = "dev",
    group_cols: tuple[str, ...] = ("lab", "period"),
) -> tuple[pd.DataFrame, str]:
    """Add per-image column centered within each lab×period tile.

    mode: ``dev`` (raw count − tile mean), ``zscore``, ``pct`` (percent of tile mean).
    """
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not in dataframe")
    if mode not in {"dev", "zscore", "pct"}:
        raise ValueError(f"mode must be 'dev', 'zscore', or 'pct'; got {mode!r}")

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    grouped = work.groupby(list(group_cols), dropna=False)[value_col]
    tile_mean = grouped.transform("mean")
    suffix = {"dev": "tile_dev", "zscore": "tile_z", "pct": "tile_pct_dev"}[mode]
    out_col = f"{value_col}_{suffix}"

    if mode == "dev":
        work[out_col] = work[value_col] - tile_mean
    elif mode == "zscore":
        tile_std = grouped.transform("std")
        work[out_col] = (work[value_col] - tile_mean) / tile_std
    else:
        work[out_col] = np.where(
            tile_mean != 0,
            100.0 * (work[value_col] - tile_mean) / tile_mean,
            np.nan,
        )
    return work, out_col


def tile_dispersion_table(
    df: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...] = ("lab", "period"),
) -> pd.DataFrame:
    """Per-tile mean, std, variance, and CV across detector positions."""
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not in dataframe")
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    out = (
        work.groupby(list(group_cols), dropna=False)[value_col]
        .agg(n="count", mean="mean", std="std", var="var")
        .reset_index()
    )
    out["cv"] = out["std"] / out["mean"]
    return out


def pivot_lab_period_dispersion(
    df: pd.DataFrame,
    value_col: str,
    stat: str = "std",
    group_cols: tuple[str, ...] = ("lab", "period"),
) -> pd.DataFrame:
    """Pivot lab × period matrix of within-tile dispersion (std, var, cv, mean)."""
    allowed = {"std", "var", "cv", "mean"}
    if stat not in allowed:
        raise ValueError(f"stat must be one of {allowed}; got {stat!r}")
    table = tile_dispersion_table(df, value_col, group_cols=group_cols)
    pivot = table.pivot(index="lab", columns="period", values=stat)
    return pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))


def plot_lab_period_scalar_heatmaps(
    pivot: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
    cmap: str = "mako_r",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    annotate_fmt: str = "{:.0f}",
):
    """Single-value-per-tile lab × period heatmap with optional cell annotations."""
    values = pivot.values.astype(float)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    fig, ax = plt.subplots(figsize=(2.8 * len(pivot.columns), 0.55 * len(pivot.index) + 2))
    norm = None
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    try:
        plot_cmap = plt.colormaps[cmap].copy()
    except KeyError:
        import seaborn as sns

        plot_cmap = sns.color_palette(cmap, as_cmap=True)
    plot_cmap.set_bad(color="#e0e0e0")
    im = ax.imshow(
        values,
        aspect="auto",
        cmap=plot_cmap,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
    )
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("period")
    ax.set_ylabel("lab")
    ax.set_title(title)

    for i, lab in enumerate(pivot.index):
        for j, period in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                mid = center if center is not None else (vmin + vmax) / 2
                text_color = "white" if val > mid else "black"
                ax.text(j, i, annotate_fmt.format(val), ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tile_dispersion_bars(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    stat_col: str,
    label_a: str,
    label_b: str,
    title: str,
    save_path: Path | None = None,
):
    """Grouped bar chart comparing per-tile dispersion between two metrics."""
    merged = table_a.merge(
        table_b,
        on=["lab", "period"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    merged["tile"] = merged["lab"] + " " + merged["period"]
    merged = merged.sort_values(["lab", "period"])

    x = np.arange(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(merged) * 0.55), 5))
    ax.bar(x - width / 2, merged[f"{stat_col}_a"], width, label=label_a, color="#4C72B0")
    ax.bar(x + width / 2, merged[f"{stat_col}_b"], width, label=label_b, color="#DD8452")
    ax.set_xticks(x, merged["tile"], rotation=45, ha="right")
    ax.set_ylabel(stat_col)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_deviation_strip(
    df: pd.DataFrame,
    dev_col: str,
    title: str,
    save_path: Path | None = None,
    hue_col: str = "position",
    order: list | None = None,
):
    """Strip plot of per-image tile-centered deviations across positions."""
    import seaborn as sns

    work = df.copy()
    work[dev_col] = pd.to_numeric(work[dev_col], errors="coerce")
    if order is None:
        order = sorted(work[hue_col].dropna().unique(), key=lambda p: (str(p)[0], int(str(p)[1:])))

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.stripplot(
        data=work,
        x=hue_col,
        y=dev_col,
        order=order,
        hue="lab",
        dodge=False,
        alpha=0.55,
        jitter=0.25,
        size=4,
        ax=ax,
    )
    ax.axhline(0, color="0.3", linewidth=1, linestyle="--")
    ax.set_xlabel("detector position")
    ax.set_ylabel(dev_col)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="lab")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def export_gmm_cref_results(
    per_image_df: pd.DataFrame,
    output_dir: Path,
    inventory_total: int | None = None,
) -> pd.DataFrame:
    """Enrich, save per-image master CSV, count comparison, and GMM summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = enrich_with_cref_metadata(per_image_df)
    enriched.to_csv(output_dir / "gmm_per_image_cref.csv", index=False)
    enriched.to_csv(output_dir / "gmm_count_comparison.csv", index=False)
    summaries = build_cref_summaries(enriched, GMM_COUNT_COLS)
    save_cref_summaries(summaries, output_dir, prefix="gmm")
    n = len(enriched)
    if inventory_total is not None and n != inventory_total:
        print(f"Note: {n} images analyzed ({inventory_total} in full CREF inventory).")
    print(f"Saved gmm_per_image_cref.csv ({n} rows) and summaries/gmm_*.csv")
    return enriched


def export_radon_cref_results(
    per_image_df: pd.DataFrame,
    output_dir: Path,
    inventory_total: int | None = None,
) -> pd.DataFrame:
    """Enrich, save per-image master CSV and radon summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = enrich_with_cref_metadata(per_image_df)
    enriched.to_csv(output_dir / "radon_per_image_cref.csv", index=False)
    summaries = build_cref_summaries(enriched, RADON_COUNT_COLS)
    save_cref_summaries(summaries, output_dir, prefix="radon")
    n = len(enriched)
    if inventory_total is not None and n != inventory_total:
        print(f"Note: {n} images analyzed ({inventory_total} in full CREF inventory).")
    print(f"Saved radon_per_image_cref.csv ({n} rows) and summaries/radon_*.csv")
    return enriched


# =====================================================================
# Configuration and Hyperparameters
# =====================================================================
@dataclass
class Config:
    """Paths, ROI, and physical hyperparameters."""
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data_cref" / "data"
    )
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs")
    image_glob: str = "CREF_*.jpg"

    filter_lab: str | None = None
    filter_period: int | None = None
    filter_position_group: str | None = None

    roi_x: int = 2188
    roi_y: int = 2244
    roi_w: int = 5072
    roi_h: int = 4960

    intensity_threshold: int = 0
    closing_kernel_size: int = 1

    min_area: float = 20.0   # 20 pixels
    max_area: float = 10000.0

    n_clusters: int = 4
    random_state: int = 42

    save_debug_images: bool = True

    # Post-GMM watershed
    ws_enabled: bool = True
    ws_min_area: float = 160.0
    ws_min_fragment_area: float = 20.0
    ws_cluster_filter: bool = True
    ws_clusters: list = field(default_factory=list)
    ws_method: str = "distance_peaks"
    ws_peak_min_distance: int = 3
    ws_h_minima: float = 2.0

    # Artifact tagging (saliency = area * max_intensity)
    artifact_saliency_ratio: float = 0.5
    artifact_clusters: list | None = None

    # Visual comparison ROI (None => first discovered image)
    visual_roi_image: str | None = None
    visual_roi_x0: int = 700
    visual_roi_y0: int = 1400
    visual_roi_size: int = 700

    @property
    def macro_summary(self) -> Path:
        return self.data_dir / "Summary.csv"


CFG = Config()
CLUSTER_CMAP = "inferno"


def cluster_norm(n: int) -> Normalize:
    return Normalize(vmin=0, vmax=max(n - 1, 1))


def cluster_colors_rgb(n: int, cmap_name: str = CLUSTER_CMAP) -> list[tuple[float, float, float]]:
    """One RGB per cluster id, sampled evenly along a continuous colormap."""
    if n <= 0:
        return []
    cmap = plt.colormaps[cmap_name]
    norm = cluster_norm(n)
    return [cmap(norm(c))[:3] for c in range(n)]


def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        x, y, w, h = cv2.boundingRect(cnt)
        return x + w // 2, y + h // 2
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def contour_max_intensity(roi, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    sub = roi[y : y + h, x : x + w]
    mask = np.zeros(sub.shape, np.uint8)
    cv2.drawContours(mask, [cnt - [[x, y]]], -1, 1, thickness=cv2.FILLED)
    vals = sub[mask.astype(bool)]
    return float(vals.max()) if vals.size else 0.0


def features_from_contour(cnt):
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = 4.0 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0
    _x, _y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0.0
    return area, perimeter, circularity, aspect_ratio


def meta_row_from_contour(image_name, roi, cnt):
    cx, cy = contour_centroid(cnt)
    area, perimeter, circularity, aspect_ratio = features_from_contour(cnt)
    return {
        "image": image_name,
        "cx": cx,
        "cy": cy,
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "max_intensity": contour_max_intensity(roi, cnt),
    }


def contour_region_mask(roi, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [cnt - [[x, y]]], -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool), (x, y)


def watershed_split_mask(region_mask, method, cfg: Config):
    dist = ndi.distance_transform_edt(region_mask)
    min_dist = cfg.ws_peak_min_distance

    if method == "distance_peaks":
        coords = peak_local_max(
            dist, min_distance=min_dist, labels=region_mask, exclude_border=False
        )
        if len(coords) <= 1:
            out = np.zeros(region_mask.shape, dtype=np.int32)
            out[region_mask] = 1
            return out
        mk = np.zeros(region_mask.shape, np.int32)
        for j, (r, c) in enumerate(coords, start=1):
            mk[r, c] = j
        mk, _ = ndi.label(mk > 0)
        return segmentation.watershed(-dist, mk, mask=region_mask)

    if method == "h_minima":
        seeds = h_minima(dist, h=cfg.ws_h_minima)
        mk, n = ndi.label(seeds)
        if n <= 1:
            out = np.zeros(region_mask.shape, dtype=np.int32)
            out[region_mask] = 1
            return out
        return segmentation.watershed(-dist, mk, mask=region_mask)

    if method == "cv2_marker":
        coords = peak_local_max(
            dist, min_distance=min_dist, labels=region_mask, exclude_border=False
        )
        if len(coords) <= 1:
            out = np.zeros(region_mask.shape, dtype=np.int32)
            out[region_mask] = 1
            return out
        markers = np.zeros(region_mask.shape, np.int32)
        for j, (r, c) in enumerate(coords, start=1):
            markers[r, c] = j
        dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dist_rgb = cv2.cvtColor(dist_u8, cv2.COLOR_GRAY2BGR)
        cv2.watershed(dist_rgb, markers)
        ws = markers.copy()
        ws[ws == -1] = 0
        ws[~region_mask] = 0
        out = np.zeros(region_mask.shape, dtype=np.int32)
        lab = 0
        for val in np.unique(ws):
            if val <= 0:
                continue
            lab += 1
            out[ws == val] = lab
        if lab == 0:
            out[region_mask] = 1
        return out

    raise ValueError(f"Unknown watershed method: {method}")


def contours_from_labeled(labeled, offset_xy, min_area):
    x0, y0 = offset_xy
    out = []
    for lab in range(1, int(labeled.max()) + 1):
        frag = (labeled == lab).astype(np.uint8)
        if int(frag.sum()) < min_area:
            continue
        cnts, _ = cv2.findContours(frag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            shifted = cnt.copy()
            shifted[:, 0, 0] += x0
            shifted[:, 0, 1] += y0
            out.append(shifted)
    return out


def split_contour_watershed(roi, cnt, method, cfg: Config):
    mask, offset = contour_region_mask(roi, cnt)
    labeled = watershed_split_mask(mask, method, cfg)
    frags = contours_from_labeled(labeled, offset, cfg.ws_min_fragment_area)
    if len(frags) <= 1:
        return [cnt]
    return frags


def should_watershed_split(area, cluster, cfg: Config, ws_clusters):
    if area < cfg.ws_min_area:
        return False
    if not cfg.ws_cluster_filter:
        return True
    return int(cluster) in ws_clusters


def resolve_ws_clusters(meta, labels, cfg: Config, ws_clusters=None):
    if ws_clusters is not None and len(ws_clusters) > 0:
        return list(ws_clusters)
    if not cfg.ws_cluster_filter:
        return []
    prof = meta.copy()
    prof["cluster"] = labels
    mean_area = prof.groupby("cluster")["area"].mean()
    return mean_area[mean_area >= cfg.ws_min_area].index.astype(int).tolist()


def apply_watershed_pass(image_registry, meta, labels, gmm, scaler, cfg: Config, ws_clusters=None):
    ws_clusters = resolve_ws_clusters(meta, labels, cfg, ws_clusters)
    new_rows = []
    n_parents_split = 0
    n_new_fragments = 0

    for reg in image_registry:
        roi = reg["roi"]
        img_name = reg["path"].name
        start = reg["label_start"]
        counts = reg["counts"]
        contours = reg["contours"]

        for local_i in range(counts):
            gi = start + local_i
            cnt = contours[local_i]
            cluster = int(labels[gi])
            area = float(meta.iloc[gi]["area"])

            if cfg.ws_enabled and should_watershed_split(area, cluster, cfg, ws_clusters):
                frags = split_contour_watershed(roi, cnt, cfg.ws_method, cfg)
                if len(frags) > 1:
                    n_parents_split += 1
                    n_new_fragments += len(frags)
                    for fc in frags:
                        new_rows.append(meta_row_from_contour(img_name, roi, fc))
                    continue
            new_rows.append(meta.iloc[gi].to_dict())

    meta_ws = pd.DataFrame(new_rows).reset_index(drop=True)
    X_ws = meta_ws[["area", "perimeter", "circularity", "aspect_ratio"]].to_numpy(dtype=np.float64)
    labels_ws = gmm.predict(scaler.transform(X_ws))
    stats = {
        "ws_clusters": ws_clusters,
        "n_parents_split": n_parents_split,
        "n_new_fragments": n_new_fragments,
        "n_before": len(meta),
        "n_after": len(meta_ws),
    }
    return meta_ws, X_ws, labels_ws, stats


def compare_watershed_methods(roi, contours, labels_for_contours, cfg: Config, ws_clusters):
    results = {}
    candidates = [
        (i, cnt)
        for i, cnt in enumerate(contours)
        if should_watershed_split(
            float(cv2.contourArea(cnt)), labels_for_contours[i], cfg, ws_clusters
        )
    ]
    for method in ("distance_peaks", "h_minima", "cv2_marker"):
        total_frags = 0
        n_split = 0
        try:
            for _, cnt in candidates:
                frags = split_contour_watershed(roi, cnt, method, cfg)
                total_frags += len(frags)
                if len(frags) > 1:
                    n_split += 1
            results[method] = {
                "n_candidates": len(candidates),
                "n_split_parents": n_split,
                "n_total_fragments": total_frags,
                "error": None,
            }
        except Exception as exc:
            results[method] = {
                "n_candidates": len(candidates),
                "n_split_parents": np.nan,
                "n_total_fragments": np.nan,
                "error": str(exc),
            }
    return results


def suggest_artifact_clusters(meta, ratio=0.5):
    prof = meta.groupby("cluster").agg(
        n=("area", "size"),
        area=("area", "mean"),
        max_int=("max_intensity", "mean"),
    )
    sal = prof["area"] * prof["max_int"]
    prof["saliency"] = sal
    med = float(sal.median()) if len(sal) else 0.0
    prof["suggest_artifact"] = sal < ratio * med if med > 0 else False
    suggested = prof.index[prof["suggest_artifact"]].astype(int).tolist()
    return prof, suggested


def load_macro_counts(summary_path: Path):
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    return dict(zip(df["Slice"], df["Count"]))


def per_image_counts(meta, is_track=None):
    sub = meta
    if is_track is not None:
        sub = meta[meta["is_track"] == is_track]
    return sub.groupby("image").size()


def build_count_comparison(macro_map, meta_gmm, meta_ws, artifact_clusters):
    gmm_counts = per_image_counts(meta_gmm)
    ws_counts = per_image_counts(meta_ws)
    clean = meta_ws[~meta_ws["cluster"].isin(artifact_clusters)]
    clean_counts = per_image_counts(clean)

    images = sorted(set(meta_gmm["image"].unique()) | set(macro_map.keys()))
    rows = []
    for img in images:
        rows.append({
            "image": img,
            "imagej_count": lookup_macro_count(macro_map, img),
            "gmm_count": int(gmm_counts.get(img, 0)),
            "gmm_ws_count": int(ws_counts.get(img, 0)),
            "gmm_ws_clean_count": int(clean_counts.get(img, 0)),
        })
    return pd.DataFrame(rows)


def roi_window_counts(meta, image, x0, y0, size, is_track=None):
    sub = meta[meta["image"] == image]
    if is_track is not None and "is_track" in sub.columns:
        sub = sub[sub["is_track"] == is_track]
    m = (
        (sub["cx"] >= x0)
        & (sub["cx"] < x0 + size)
        & (sub["cy"] >= y0)
        & (sub["cy"] < y0 + size)
    )
    return int(m.sum())


def plot_roi_variant_comparison(
    roi,
    meta,
    image,
    x0,
    y0,
    size,
    colors_rgb,
    title="",
    is_track=None,
    ax=None,
):
    crop = roi[y0 : y0 + size, x0 : x0 + size]
    stretch = np.clip(crop.astype(np.float32) * 8, 0, 255).astype(np.uint8)
    sub = meta[meta["image"] == image]
    if is_track is not None and "is_track" in sub.columns:
        sub = sub[sub["is_track"] == is_track]
    m = (
        (sub["cx"] >= x0)
        & (sub["cx"] < x0 + size)
        & (sub["cy"] >= y0)
        & (sub["cy"] < y0 + size)
    )
    cands = sub.loc[m]
    created = ax is None
    if created:
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(stretch, cmap="gray")
    ax[0].set_title("ROI stretch 8×")
    ax[0].axis("off")
    ax[1].imshow(stretch, cmap="gray")
    for _, r in cands.iterrows():
        c = int(r["cluster"]) if "cluster" in r else 0
        ax[1].scatter(
            r["cx"] - x0,
            r["cy"] - y0,
            s=18,
            c=[colors_rgb[c % len(colors_rgb)]],
            edgecolors="white",
            linewidths=0.3,
        )
    ax[1].set_title(f"markers (n={len(cands)})")
    ax[1].axis("off")
    if title:
        ax[0].figure.suptitle(title, fontsize=10)
    if created:
        plt.tight_layout()
    return len(cands), ax


def plot_roi_three_variants(
    roi,
    meta_gmm,
    meta_ws,
    image,
    x0,
    y0,
    size,
    colors_rgb,
    save_path=None,
):
    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    panels = [
        ("GMM", meta_gmm, None),
        ("GMM + WS", meta_ws, None),
        ("GMM + WS clean", meta_ws, True),
    ]
    counts = {}
    for row, (name, meta, track_only) in enumerate(panels):
        is_track = track_only if track_only is not None else None
        if is_track is True:
            sub_meta = meta_ws[meta_ws["is_track"]].copy()
        else:
            sub_meta = meta
        n, _ = plot_roi_variant_comparison(
            roi, sub_meta, image, x0, y0, size, colors_rgb, ax=axes[row]
        )
        counts[name] = n
        axes[row, 0].set_ylabel(name, fontsize=9)

    fig.suptitle(
        f"{image} @ ({x0},{y0}) {size}×{size} — GMM / GMM+WS / GMM+WS clean",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return counts, fig


def extract_features_from_roi(image_path: Path, cfg: Config):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Error: Failed to load {image_path.name}")
        return None, None, None, None

    h, w = img.shape
    roi_y2 = min(cfg.roi_y + cfg.roi_h, h)
    roi_x2 = min(cfg.roi_x + cfg.roi_w, w)

    if cfg.roi_y >= h or cfg.roi_x >= w:
        print(f"  Warning: ROI out of bounds for {image_path.name}.")
        return None, None, None, None

    roi = img[cfg.roi_y:roi_y2, cfg.roi_x:roi_x2]

    _, binary_signal = cv2.threshold(roi, cfg.intensity_threshold, 255, cv2.THRESH_BINARY)
    weld_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.closing_kernel_size, cfg.closing_kernel_size)
    )
    welded_signal = cv2.morphologyEx(binary_signal, cv2.MORPH_CLOSE, weld_kernel)

    contours, _ = cv2.findContours(welded_signal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    valid_contours = []

    for cnt in contours:
        area, perimeter, circularity, aspect_ratio = features_from_contour(cnt)
        if area < cfg.min_area or area > cfg.max_area:
            continue
        if perimeter == 0:
            continue
        features.append([area, perimeter, circularity, aspect_ratio])
        valid_contours.append(cnt)

    if not features:
        return np.empty((0, 4)), [], roi, welded_signal

    return np.asarray(features, dtype=np.float64), valid_contours, roi, welded_signal


def extract_dataset(image_paths, cfg: Config):
    all_features = []
    meta_rows = []
    image_registry = []
    label_offset = 0

    for path in image_paths:
        feats, cnts, roi, debug_thresh = extract_features_from_roi(path, cfg)
        if feats is None:
            continue
        if cfg.save_debug_images and debug_thresh is not None:
            cv2.imwrite(str(cfg.output_dir / f"debug_binary_{path.stem}.png"), debug_thresh)
        if len(feats) == 0:
            continue
        for cnt in cnts:
            meta_rows.append(meta_row_from_contour(path.name, roi, cnt))
        all_features.append(feats)
        image_registry.append({
            "path": path,
            "counts": len(feats),
            "label_start": label_offset,
            "contours": cnts,
            "roi": roi,
        })
        label_offset += len(feats)

    if not all_features:
        return None, None, None, None
    X = np.vstack(all_features)
    meta = pd.DataFrame(meta_rows)
    return X, meta, image_registry, image_paths


def save_count_comparison_plots(final_df, output_dir, colors_rgb=None):
    x = np.arange(len(final_df))
    w = 0.2
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5 * w, final_df["imagej_count"], w, label="ImageJ macro")
    ax.bar(x - 0.5 * w, final_df["gmm_count"], w, label="GMM")
    ax.bar(x + 0.5 * w, final_df["gmm_ws_count"], w, label="GMM + WS")
    ax.bar(x + 1.5 * w, final_df["gmm_ws_clean_count"], w, label="GMM + WS − artifacts")
    ax.set_xticks(x)
    ax.set_xticklabels([image_display_name(n) for n in final_df["image"]], rotation=90)
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Per-image count comparison")
    plt.tight_layout()
    fig.savefig(output_dir / "gmm_count_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Unsupervised GMM clustering for CREF CR-39 scans.")
    parser.add_argument("--lab", default=None, help="Filter by lab code, e.g. P2-DOT")
    parser.add_argument("--period", type=int, choices=[1, 2, 3], default=None, help="Filter by period 1–3 (T1–T3)")
    parser.add_argument(
        "--position",
        choices=["A", "B", "C"],
        default=None,
        help="Filter by detector position group A, B, or C",
    )
    return parser.parse_args()


def main():
    cfg = CFG
    args = _parse_cli_args()
    if args.lab is not None:
        cfg.filter_lab = args.lab
    if args.period is not None:
        cfg.filter_period = args.period
    if args.position is not None:
        cfg.filter_position_group = args.position

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    inventory = summarize_cref_inventory(cfg.data_dir)
    print(f"CREF inventory: {inventory['total_files']} files in {cfg.data_dir}")
    print(f"  labs: {inventory['labs']}")
    print(f"  periods: {inventory['periods']}")
    print(f"  position groups: {inventory['position_groups']}")

    image_paths = discover_images(cfg)
    if not image_paths:
        print(f"No files matching {cfg.image_glob} with current filters in {cfg.data_dir}")
        return
    print(
        f"Selected {len(image_paths)} image(s)"
        f" (lab={cfg.filter_lab}, period={cfg.filter_period}, position={cfg.filter_position_group})"
    )

    print("Extracting physical features with spatial coalescence (welding)...")
    extracted = extract_dataset(image_paths, cfg)
    if extracted[0] is None:
        print("CRITICAL ZERO COUNT: The signal is completely empty.")
        return
    X, meta, image_registry, _ = extracted
    print(f"\nTotal physical events across {len(image_registry)} images: {len(X)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Fitting Gaussian Mixture Model with k={cfg.n_clusters} classes...")
    gmm = GaussianMixture(
        n_components=cfg.n_clusters, covariance_type="full", random_state=cfg.random_state
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    meta_gmm = meta.copy()
    meta_gmm["cluster"] = labels

    colors_rgb = cluster_colors_rgb(cfg.n_clusters, CLUSTER_CMAP)
    cluster_cmap = plt.colormaps[CLUSTER_CMAP]
    c_norm = cluster_norm(cfg.n_clusters)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X[:, 0], X[:, 2], c=labels, cmap=cluster_cmap, norm=c_norm, alpha=0.65, s=8,
    )
    plt.xlabel("Track Area (pixels)")
    plt.ylabel(r"Circularity Index ($4\pi A / P^2$)")
    plt.title("GMM Clustering of Welded CR-39 Alpha Tracks")
    plt.colorbar(scatter, label="GMM Cluster", ticks=list(range(cfg.n_clusters)))
    plt.grid(True, alpha=0.3)
    plt.savefig(cfg.output_dir / "fig1_feature_space.pdf", bbox_inches="tight")
    plt.close()

    profile_rows = []
    for cluster_id in range(cfg.n_clusters):
        mask = labels == cluster_id
        if mask.any():
            cluster_feats = X[mask]
            profile_rows.append({
                "cluster_id": cluster_id,
                "event_count": int(mask.sum()),
                "mean_area_px": float(np.mean(cluster_feats[:, 0])),
                "mean_circularity": float(np.mean(cluster_feats[:, 2])),
                "mean_aspect_ratio": float(np.mean(cluster_feats[:, 3])),
            })
    pd.DataFrame(profile_rows).to_csv(cfg.output_dir / "gmm_cluster_profiles.csv", index=False)

    if image_registry:
        sample = image_registry[0]
        start = sample["label_start"]
        sample_labels = labels[start : start + sample["counts"]]
        roi_norm = cv2.normalize(sample["roi"], None, 0, 255, cv2.NORM_MINMAX)
        roi_color = cv2.cvtColor(roi_norm, cv2.COLOR_GRAY2BGR)
        colors_bgr = [(int(b * 255), int(g * 255), int(r * 255)) for r, g, b in colors_rgb]
        for idx, cnt in enumerate(sample["contours"]):
            cv2.drawContours(roi_color, [cnt], -1, colors_bgr[int(sample_labels[idx])], 1)
        plt.figure(figsize=(12, 12), dpi=150)
        plt.imshow(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Segmented Morphology - {sample['path'].name}")
        plt.axis("off")
        plt.savefig(cfg.output_dir / "fig2_spatial_mapping.pdf", bbox_inches="tight")
        plt.close()

    ws_clusters = cfg.ws_clusters if cfg.ws_clusters else None
    meta_ws, X_ws, labels_ws, ws_stats = apply_watershed_pass(
        image_registry, meta_gmm, labels, gmm, scaler, cfg, ws_clusters
    )
    meta_ws = meta_ws.copy()
    meta_ws["cluster"] = labels_ws
    print(
        f"Watershed ({cfg.ws_method}): clusters={ws_stats['ws_clusters']}  "
        f"split_parents={ws_stats['n_parents_split']}  "
        f"{ws_stats['n_before']} -> {ws_stats['n_after']} events"
    )

    meta_ws["saliency"] = meta_ws["area"] * meta_ws["max_intensity"]
    prof, suggested = suggest_artifact_clusters(meta_ws, cfg.artifact_saliency_ratio)
    prof.to_csv(cfg.output_dir / "gmm_cluster_saliency_profiles.csv")
    artifact_clusters = (
        cfg.artifact_clusters if cfg.artifact_clusters is not None else suggested
    )
    meta_ws["is_track"] = ~meta_ws["cluster"].isin(artifact_clusters)
    print(f"ARTIFACT_CLUSTERS = {artifact_clusters}")

    macro_map = load_macro_counts(cfg.macro_summary)
    final_df = build_count_comparison(macro_map, meta_gmm, meta_ws, artifact_clusters)
    final_df = export_gmm_cref_results(
        final_df, cfg.output_dir, inventory_total=inventory["total_files"]
    )
    save_count_comparison_plots(final_df, cfg.output_dir)

    roi_lookup = {reg["path"].name: reg["roi"] for reg in image_registry}
    visual_image = resolve_visual_roi_image(cfg, list(roi_lookup.keys()))
    if visual_image in roi_lookup:
        roi_stem = Path(visual_image).stem
        counts, fig = plot_roi_three_variants(
            roi_lookup[visual_image],
            meta_gmm,
            meta_ws,
            visual_image,
            cfg.visual_roi_x0,
            cfg.visual_roi_y0,
            cfg.visual_roi_size,
            colors_rgb,
            save_path=cfg.output_dir / f"gmm_roi_compare_{roi_stem}.png",
        )
        plt.close(fig)
        print(f"Visual ROI counts ({visual_image}): {counts}")

    joblib.dump({"scaler": scaler, "gmm": gmm}, cfg.output_dir / "gmm_model.joblib")
    meta_gmm.to_csv(cfg.output_dir / "meta_gmm.csv", index=False)
    meta_ws.to_csv(cfg.output_dir / "meta_gmm_ws.csv", index=False)
    print(f"Pipeline executed successfully. Outputs in '{cfg.output_dir}'")


if __name__ == "__main__":
    main()
