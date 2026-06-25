"""
Unsupervised GMM clustering for ultra-faint Darkfield CR-39 track scans.
Implements spatial welding (Morphological Closing) to reconstruct
fragmented track signals before area filtering, optional post-GMM watershed
splitting, saliency-based artifact tagging, and count comparison vs ImageJ.
"""

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
# Configuration and Hyperparameters
# =====================================================================
@dataclass
class Config:
    """Paths, ROI, and physical hyperparameters."""
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data_test")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs")
    image_glob: str = "LBS*.jpg"

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

    # Visual comparison ROI
    visual_roi_image: str = "LBS255611.jpg"
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
            "imagej_count": macro_map.get(img, np.nan),
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
    ax.set_xticklabels([n[-7:] for n in final_df["image"]], rotation=90)
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Per-image count comparison")
    plt.tight_layout()
    fig.savefig(output_dir / "gmm_count_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = CFG
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(cfg.data_dir.glob(cfg.image_glob))
    if not image_paths:
        print(f"Directory empty or path incorrect. No files matching {cfg.image_glob}")
        return

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
    final_df.to_csv(cfg.output_dir / "gmm_count_comparison.csv", index=False)
    save_count_comparison_plots(final_df, cfg.output_dir)

    roi_lookup = {reg["path"].name: reg["roi"] for reg in image_registry}
    if cfg.visual_roi_image in roi_lookup:
        counts, fig = plot_roi_three_variants(
            roi_lookup[cfg.visual_roi_image],
            meta_gmm,
            meta_ws,
            cfg.visual_roi_image,
            cfg.visual_roi_x0,
            cfg.visual_roi_y0,
            cfg.visual_roi_size,
            colors_rgb,
            save_path=cfg.output_dir / "gmm_roi_compare_LBS255611.png",
        )
        plt.close(fig)
        print(f"Visual ROI counts: {counts}")

    joblib.dump({"scaler": scaler, "gmm": gmm}, cfg.output_dir / "gmm_model.joblib")
    meta_gmm.to_csv(cfg.output_dir / "meta_gmm.csv", index=False)
    meta_ws.to_csv(cfg.output_dir / "meta_gmm_ws.csv", index=False)
    print(f"Pipeline executed successfully. Outputs in '{cfg.output_dir}'")


if __name__ == "__main__":
    main()
