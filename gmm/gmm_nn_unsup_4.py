"""
Unsupervised GMM clustering for ultra-faint Darkfield CR-39 track scans.
Implements spatial welding (Morphological Closing) to reconstruct 
fragmented track signals before area filtering.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
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

    # ROI Definition (x, y, width, height)
    roi_x: int = 2188
    roi_y: int = 2244
    roi_w: int = 5072
    roi_h: int = 4960

    # Hard global threshold: 0 means anything strictly > 0 is captured
    intensity_threshold: int = 0
    
    # Kernel size to weld fragmented faint pixels back into single tracks
    # 3x3 or 5x5 is typically optimal for resolving JPEG artifacts on 5000x5000 scans
    # 7x7 is more inclusive of the tracks (since it includes the entire track area)
    # In the context of ultra-faint darkfield microscopy, 
    # the closing_kernel_size parameter defines the spatial dimensions of the structuring element 
    # (typically an elliptical matrix) utilized during the morphological closing operation. 
    # Due to the extremely low signal-to-noise ratio inherent to these 
    # scans—where track intensities approach the sensor's quantization 
    # limit—and the presence of lossy compression artifacts, 
    # a single physical alpha-particle impact site is frequently digitized 
    # as a cluster of disjointed pixel fragments. 
    # Morphological closing, which consists of a dilation followed by an erosion,
    #  acts as a spatial coalescing filter. It bridges the artificial zero-intensity 
    # micro-gaps between adjacent signal pixels without artificially 
    # inflating the macroscopic boundary or altering the invariant geometric features 
    # (such as circularity) of the original object.We implemented this parameter as a 
    # targeted "welding" mechanism immediately following the global intensity thresholding 
    # and strictly prior to the contour extraction phase. 
    # By applying a carefully tuned closing kernel (e.g., $5 \times 5$ pixels), 
    # the fragmented components of a primary track are reconstructed into a single, 
    # contiguous morphological entity.
    # This spatial reconstruction is critical for the survival of the signal 
    # through the subsequent analytical pipeline; without it, 
    # genuine tracks would be erroneously interpreted as independent sub-components, 
    # failing the minimum acceptance threshold (e.g., A < 20 px) and being discarded as thermal noise. 
    # Consequently, the kernel size acts as the fundamental bridge between the optical 
    # digitization process and the physical integrity of the spectrometric counting.
    closing_kernel_size: int = 7
    
    # Physical constraints
    min_area: float = 20.0
    max_area: float = 10000.0
    
    # GMM clustering
    n_clusters: int = 8
    random_state: int = 42
    
    # Output control
    save_debug_images: bool = True

CFG = Config()

# Default vivid colormap for cluster overlays (scales to any n_clusters)
CLUSTER_CMAP = "turbo"


def cluster_colors_rgb(n: int, cmap_name: str = CLUSTER_CMAP) -> list[tuple[float, float, float]]:
    """Return *n* distinct vivid RGB colours sampled from a matplotlib colormap."""
    if n <= 0:
        return []
    cmap = plt.colormaps[cmap_name]
    if getattr(cmap, "N", None) and n <= cmap.N:
        return [cmap(i)[:3] for i in range(n)]
    if n == 1:
        return [cmap(0.5)[:3]]
    return [cmap(i / (n - 1))[:3] for i in range(n)]


def extract_features_from_roi(image_path: Path, cfg: Config):
    """
    Reads image, isolates ROI, captures all non-zero pixels, 
    welds fragmented tracks, and computes geometric features.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Error: Failed to load {image_path.name}")
        return None, None, None, None

    # Safe crop to boundaries
    h, w = img.shape
    roi_y2 = min(cfg.roi_y + cfg.roi_h, h)
    roi_x2 = min(cfg.roi_x + cfg.roi_w, w)
    
    if cfg.roi_y >= h or cfg.roi_x >= w:
        print(f"  Warning: ROI out of bounds for {image_path.name}.")
        return None, None, None, None

    roi = img[cfg.roi_y:roi_y2, cfg.roi_x:roi_x2]
    
    # 1. Direct Absolute Thresholding
    # NO BLURRING. Blurring destroys intensity=1 pixels when surrounded by 0s.
    # Any pixel strictly greater than intensity_threshold (0) becomes 255.
    _, binary_signal = cv2.threshold(roi, cfg.intensity_threshold, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological Closing (Welding)
    # This step bridges 1-to-2 pixel gaps of pure black that split a single 
    # physical track into multiple small, sub-threshold fragments.
    weld_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.closing_kernel_size, cfg.closing_kernel_size))
    welded_signal = cv2.morphologyEx(binary_signal, cv2.MORPH_CLOSE, weld_kernel)
    
    # 3. Contour Extraction
    contours, _ = cv2.findContours(welded_signal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Apply the Area bandpass filter on the WELDED tracks
        if area < cfg.min_area or area > cfg.max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Circularity invariant: 4 * pi * Area / Perimeter^2
        circularity = 4 * np.pi * area / (perimeter**2)
        
        # Aspect Ratio
        _x, _y, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box) / h_box if h_box > 0 else 0.0

        features.append([area, perimeter, circularity, aspect_ratio])
        valid_contours.append(cnt)

    if not features:
        return np.empty((0, 4)), [], roi, welded_signal

    return np.asarray(features, dtype=np.float64), valid_contours, roi, welded_signal

def main():
    cfg = CFG
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(cfg.data_dir.glob(cfg.image_glob))
    if not image_paths:
        print(f"Directory empty or path incorrect. No files matching {cfg.image_glob}")
        return

    print("Extracting physical features with spatial coalesence (welding)...")
    all_features = []
    image_registry = []
    label_offset = 0

    for path in image_paths:
        print(f"  Scanning {path.name}...")
        feats, cnts, roi, debug_thresh = extract_features_from_roi(path, cfg)
        
        if feats is None:
            continue
            
        print(f"    -> Extracted {len(feats)} valid events (Area >= {cfg.min_area}px).")
        
        if cfg.save_debug_images and debug_thresh is not None:
            debug_path = cfg.output_dir / f"debug_binary_{path.stem}.png"
            cv2.imwrite(str(debug_path), debug_thresh)

        if len(feats) > 0:
            all_features.append(feats)
            image_registry.append({
                "path": path,
                "counts": len(feats),
                "label_start": label_offset,
                "contours": cnts,
                "roi": roi
            })
            label_offset += len(feats)

    if not all_features:
        print("CRITICAL ZERO COUNT: The signal is completely empty.")
        return

    X = np.vstack(all_features)
    print(f"\nTotal physical events accumulated across {len(image_registry)} images: {len(X)}")
    
    # Standardize data for GMM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Fitting Gaussian Mixture Model with k={cfg.n_clusters} classes...")
    gmm = GaussianMixture(n_components=cfg.n_clusters, covariance_type='full', random_state=cfg.random_state)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    # =================================================================
    # Visualization and Exports
    # =================================================================
    
    # Fig 1: Feature Space
    from matplotlib.colors import BoundaryNorm, ListedColormap
    colors_rgb = cluster_colors_rgb(cfg.n_clusters)
    discrete_cmap = ListedColormap(colors_rgb)
    bound_norm = BoundaryNorm(np.arange(cfg.n_clusters + 1) - 0.5, cfg.n_clusters)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X[:, 0], X[:, 2], c=labels, cmap=discrete_cmap, norm=bound_norm, alpha=0.65, s=8,
    )
    plt.xlabel('Track Area (pixels)')
    plt.ylabel(r'Circularity Index ($4\pi A / P^2$)')
    plt.title('GMM Clustering of Welded CR-39 Alpha Tracks')
    plt.colorbar(scatter, label='GMM Cluster', ticks=list(range(cfg.n_clusters)))
    plt.grid(True, alpha=0.3)
    plt.savefig(cfg.output_dir / 'fig1_feature_space.pdf', bbox_inches='tight')
    plt.close()

    # Cluster Profiles
    profile_rows = []
    for cluster_id in range(cfg.n_clusters):
        mask = labels == cluster_id
        cluster_feats = X[mask]
        if len(cluster_feats) > 0:
            profile_rows.append({
                "cluster_id": cluster_id,
                "event_count": len(cluster_feats),
                "mean_area_px": float(np.mean(cluster_feats[:, 0])),
                "mean_circularity": float(np.mean(cluster_feats[:, 2])),
                "mean_aspect_ratio": float(np.mean(cluster_feats[:, 3]))
            })
    pd.DataFrame(profile_rows).to_csv(cfg.output_dir / 'gmm_cluster_profiles.csv', index=False)

    # Fig 2: Spatial mapping 
    if len(image_registry) > 0:
        sample = image_registry[0]
        start = sample["label_start"]
        end = start + sample["counts"]
        sample_labels = labels[start:end]
        
        # Enhanced visibility for darkfield overlay
        roi_norm = cv2.normalize(sample["roi"], None, 0, 255, cv2.NORM_MINMAX)
        roi_color = cv2.cvtColor(roi_norm, cv2.COLOR_GRAY2BGR)
        
        colors_bgr = [
            (int(b * 255), int(g * 255), int(r * 255)) for r, g, b in colors_rgb
        ]
        
        for idx, cnt in enumerate(sample["contours"]):
            c_id = sample_labels[idx]
            color = colors_bgr[c_id]
            cv2.drawContours(roi_color, [cnt], -1, color, 1)
            
        plt.figure(figsize=(12, 12), dpi=150)
        plt.imshow(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        plt.title(f'Segmented Morphology - {sample["path"].name}')
        plt.axis('off')
        plt.savefig(cfg.output_dir / 'fig2_spatial_mapping.pdf', bbox_inches='tight')
        plt.close()

    joblib.dump({"scaler": scaler, "gmm": gmm}, cfg.output_dir / "gmm_model.joblib")
    print(f"Pipeline executed successfully. Outputs in '{cfg.output_dir}'")

if __name__ == "__main__":
    main()