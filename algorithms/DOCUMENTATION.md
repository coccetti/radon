# CR-39 Radon Track Analysis — Software Description and Output File Reference

## 1. Purpose

This software is a **Jupyter notebook pipeline** (`algorithms/radon_track_nn.ipynb`) that analyses scanned images of **CR-39 plastic nuclear track detectors** to count **alpha-particle tracks** produced by radon decay. It replaces and extends a simple ImageJ macro (`data_test/CR39-Scanner.txt`) that was used to batch-count tracks across many JPEG scans.

The macro approach is fast but limited: it uses a fixed brightness threshold, rejects non-circular objects, and cannot distinguish real etched tracks from dust, scratches, or merged blobs. The notebook implements a **three-stage analysis strategy**, plus an optional **validation stage**:

1. **Classical baseline** — reproduce the macro, then improve it.
2. **Cellpose-SAM** — pretrained neural instance segmentation (no training required).
3. **Unsupervised learning** — convolutional autoencoder + clustering to separate tracks from artifacts and reveal track morphology types.
4. **Ground-truth evaluation** — interactive hand-annotation on a small crop, with precision/recall/F1 for each method.

All numerical results and diagnostic figures are written to `algorithms/outputs/`.

---

## 2. Scientific and image context

### What is being measured

CR-39 detectors record the passage of charged particles (here, alpha particles from radon and its decay products) as **etched pits or bright spots** on a dark background after chemical etching and optical scanning.

Each scan is a large grayscale JPEG (~9448 × 9448 pixels). Inside the analysis region of interest (ROI), the background is essentially **black (pixel value 0)** and tracks appear as **faint bright blobs** (typically gray levels 1–56 inside the ROI; much brighter pixels exist only at the scanner border outside the ROI).

Track shape carries physics information:

- **Round tracks** — alpha particles incident roughly perpendicular to the detector surface.
- **Elongated / comet-shaped tracks** — oblique incidence angles or higher-energy alphas.
- **Merged blobs** — two or more tracks touching.
- **Faint specks** — dust, JPEG noise, or etching artifacts (not real tracks).

### What the original ImageJ macro does

For each image in a folder, the macro:

1. Opens the JPEG.
2. Sets threshold: foreground = pixels ≥ 1.
3. Crops a rectangle: `x=2188, y=2244, width=5072, height=4960`.
4. Runs **Analyze Particles** with:
   - size: 20–250 pixels
   - circularity: 0.70–1.00
5. Writes a per-image count to `Summary.csv`.

The circularity filter silently discards elongated real tracks. The fixed threshold and size window cannot split overlapping tracks or reject artifacts intelligently.

---

## 3. Software architecture

### Main program

| Component | Location | Role |
|-----------|----------|------|
| Notebook | `algorithms/radon_track_nn.ipynb` | Full pipeline: detection, NN, clustering, evaluation |
| Input images | `data_test/*.jpg` | CR-39 scan images |
| Macro reference | `data_test/CR39-Scanner.txt` | Original ImageJ logic |
| Macro output | `data_test/Summary.csv` | Baseline counts from ImageJ |
| Results | `algorithms/outputs/` | CSV tables, embeddings, figures |

### Runtime environment

Designed for the conda environment **`pytmetalbeta`** with:

- Python 3.13
- PyTorch with **Apple MPS** (Mac GPU)
- scikit-image, scikit-learn, matplotlib, pandas
- Cellpose 4.x (pretrained segmentation)
- ipympl (interactive clicking in Step 4)

Jupyter kernel name: **"Python (pytmetalbeta · MPS)"**.

### Central configuration (`Config` class)

All tunable parameters live in one dataclass at the top of the notebook:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `roi_x, roi_y, roi_w, roi_h` | 2188, 2244, 5072, 4960 | Same ROI as the ImageJ macro |
| `macro_thresh` | 1 | Foreground threshold (≥ 1) |
| `macro_size_min/max` | 20 / 250 | Macro particle size filter |
| `macro_circ_min/max` | 0.70 / 1.00 | Macro circularity filter |
| `det_fixed_thresh` | 1 | Improved detector threshold |
| `det_split_area` | 160 | Only blobs larger than this are watershed-split |
| `patch_size` | 48 | Autoencoder input patch (pixels) |
| `max_patches` | 20000 | Cap on training patches |
| `latent_dim` | 32 | Autoencoder bottleneck size |
| `ae_epochs` | 25 | Training epochs |
| `n_clusters` | 6 | KMeans cluster count |

---

## 4. Pipeline steps — detailed description

### Step 0 — Setup and image I/O

**What it does:**

- Locates all `*.jpg` files in `data_test/`.
- Loads each image as grayscale (`uint8`).
- Extracts the ROI slice matching the macro rectangle.
- Prints intensity statistics and shows histograms to justify the threshold choice.

**Key finding encoded in the software:** inside the ROI, background = 0 and tracks are very dim, so the correct foreground rule is **pixel ≥ 1**, identical to the macro's `setThreshold(1, 255)`.

---

### Step 1 — Classical baseline

Two detectors run on every image's ROI.

#### 1a. `macro_count` (reproduction)

Faithfully re-implements ImageJ **Analyze Particles** in Python:

- 8-connected components on mask `roi >= 1`
- Keeps objects with area 20–250 px and circularity 0.70–1.00
- Circularity formula: `4π × area / perimeter²` (clamped to 1.0, as in ImageJ)

Purpose: validate the Python pipeline against `data_test/Summary.csv`. Typical agreement is within ~12% of ImageJ counts (small differences come from implementation details of perimeter/area and connectivity).

#### 1b. `improved_detect` (enhanced classical)

Designed for **high recall** — find all plausible track candidates, including those the macro drops:

| Feature | Macro | Improved |
|---------|-------|----------|
| Threshold | ≥ 1 | ≥ 1 (same) |
| Circularity gate | yes (0.7–1.0) | **no** |
| Upper size limit | 250 px | effectively none |
| Touching tracks | counted as one blob | **watershed split** if area > 160 px |

Watershed splitting uses the distance transform and local maxima as seeds, applied **only to large merged blobs** so single elongated tracks are not over-split.

**Typical result:** ~33% more candidates than the macro (e.g. 795 → 1114 for one image). The overlay figure shows green markers = macro keeps, red = macro drops (mostly elongated or merged tracks).

---

### Step 2 — Cellpose-SAM baseline

**What it does:**

- Loads the pretrained **Cellpose 4** model on MPS/GPU.
- Runs instance segmentation on a representative 700×700 sub-crop of the ROI (densest region).
- Optionally (flag `RUN_CELLPOSE_FULL=False` by default) tiles the full ROI in 1024×1024 blocks.

**Preprocessing for Cellpose:**

- Contrast stretch (`×8`) because tracks are faint.
- Optional upscale (`CELLPOSE_UPSCALE=2`) to bring small tracks closer to Cellpose's training scale.

**Purpose:** a strong, zero-training alternative that natively **splits touching objects**. Compared on the same sub-crop: macro-style ~23, improved ~49, Cellpose ~7 (Cellpose is conservative on this sub-crop and may need threshold tuning via `flow_threshold` / `cellprob_threshold`).

Output: visual overlay of instance masks on the sub-crop.

---

### Step 3 — Unsupervised neural pipeline

This is the "smart" core: **no manual labels on individual tracks**, yet it separates track types and artifacts.

#### 3.1 Candidate patch extraction

- Uses detections from `improved_detect` across all images (~14,500 candidates total on the test dataset).
- For each detection centroid, crops a **48×48 pixel** patch centred on the track.
- Stores metadata per candidate: image name, position, area, eccentricity, major/minor axis, circularity, mean and max intensity.

#### 3.2 Per-patch normalisation

Each patch is divided by its own peak intensity so tracks fill `[0, 1]`. This forces the autoencoder to learn **shape and structure**, not absolute brightness (brightness is kept separately in the `max_int` feature for clustering).

#### 3.3 Convolutional autoencoder (PyTorch, MPS)

Architecture:

- **Encoder:** 3 conv layers (16→32→64 channels, stride 2) → fully connected → **32-D latent vector**
- **Decoder:** mirror with transposed convolutions → reconstructed 48×48 patch
- **Loss:** mean squared error (reconstruction)
- **Training:** 25 epochs, batch size 256, Adam optimizer

The 32-dimensional bottleneck is the **learned embedding** per candidate.

#### 3.4 Feature fusion for clustering

Two feature blocks are combined:

1. **Autoencoder embedding** (32 dims), standardised and scaled by `1/√32`
2. **Hand-crafted shape features** (6 dims): area, eccentricity, circularity, major axis, minor axis, max intensity — standardised, scaled by `1/√6`, weighted ×2

This prevents the high-dimensional AE block from drowning out interpretable morphology.

#### 3.5 Clustering

- Silhouette score computed for k = 3…8 to guide cluster count choice.
- **KMeans** with k = 6 (default).
- Produces cluster labels 0–5.

**Observed cluster semantics** (from montages; names are interpretive):

| Cluster | Typical appearance | Stats (approx.) | Role |
|---------|-------------------|-----------------|------|
| 0 | Round, medium-bright tracks | large n, moderate area | Real tracks |
| 1 | Tiny, perfectly round, max intensity = 1 | circ ≈ 1.0, maxI = 1 | **Artifact** (dust/speckle) |
| 2 | Rare outliers | very small n | Inspect manually |
| 3 | Small faint blobs | low area | Mixed / borderline |
| 4 | Tiny vertical bars, maxI = 1 | small area, high circ | **Artifact** (scratch-like) |
| 5 | Elongated comet tracks | ecc ≈ 0.76, circ ≈ 0.73 | Real oblique tracks (macro drops these) |

#### 3.6 Artifact labelling (one human decision)

Instead of labelling thousands of tracks, you label **clusters**:

- A heuristic flags faint, low-saliency clusters as artifact candidates.
- Default: `ARTIFACT_CLUSTERS = [1, 4]`
- You inspect `cluster_*.png` montages and edit this list.
- All candidates in artifact clusters get `is_track = False`; others get `is_track = True`.

#### 3.7 Final cleaned counts

Per image:

- `nn_clean_count` = number of improved candidates with `is_track = True`
- Per-cluster breakdown columns (`cluster_0`, `cluster_2`, etc.) show track-type composition.

**Typical result:** NN-cleaned counts (~420–677 per image) are **lower** than the macro (~640–829) because artifact clusters are removed, but they include elongated tracks the macro never counted.

#### 3.8 Visualisation

- **t-SNE** (after PCA) of embeddings, coloured by cluster — shows separation of morphology groups.
- **Cluster montages** — random sample of patches per cluster.
- **Mean patch per cluster** — quick fingerprint of each group's shape.

---

### Step 4 — Ground truth and precision / recall / F1

**Purpose:** turn the pipeline from visually plausible into a **validated measurement**.

#### 4.1 Evaluation crop selection

- Picks the densest 700×700 px window inside the ROI of a chosen image (`EVAL_IMG=0` by default).
- Displays the crop contrast-stretched for visibility.

#### 4.2 Interactive annotation (ipympl)

- **Left-click** → mark a real track centre (red `+`)
- **Right-click** → remove nearest mark
- Loads existing GT from CSV if present (refine without restarting)

#### 4.3 Save ground truth

Writes `gt_<image>_<x0>_<y0>_<size>.csv` with columns:

- `x_crop, y_crop` — position in crop coordinates
- `x_abs, y_abs` — position in full ROI coordinates

#### 4.4 Scoring

For each method, detections are matched to GT points by **greedy nearest-neighbour** within `MATCH_RADIUS` (default 12 px):

| Metric | Definition |
|--------|------------|
| **TP** | Detection matched to a GT point |
| **FP** | Detection with no GT nearby (over-count) |
| **FN** | GT point with no detection nearby (miss) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | Harmonic mean of precision and recall |

Methods compared:

- `macro` — true macro rules (size + circularity)
- `improved` — all improved detector candidates in the crop
- `NN_clean` — improved candidates with `is_track = True`
- `cellpose` — Cellpose masks in the crop (if Step 2 ran)

#### 4.5 TP/FP/FN overlay figure

On the evaluation crop:

- **Green circles** = true positives
- **Red circles** = false positives
- **Yellow squares** = false negatives (missed GT tracks)

---

## 5. Output files — detailed reference

All files are written to **`algorithms/outputs/`**.

### 5.1 CSV tables

#### `step1_counts.csv`

Per-image comparison of classical methods.

| Column | Type | Description |
|--------|------|-------------|
| `image` | string | JPEG filename (e.g. `LBS255611.jpg`) |
| `imagej_count` | int | Count from original `data_test/Summary.csv` |
| `macro_repro` | int | Python re-implementation of the macro |
| `improved` | int | Improved detector count (no circularity gate, watershed split) |
| `repro_err_%` | float | `100 × (macro_repro − imagej_count) / imagej_count` |
| `extra_vs_macro` | int | `improved − imagej_count` |
| `extra_%` | float | Percentage more candidates than macro |

**Use:** quantify how many tracks the macro's circularity filter silently drops.

---

#### `candidates_labeled.csv`

One row per track candidate (~14,500 rows on the test dataset). The master catalogue of all detections.

| Column | Type | Description |
|--------|------|-------------|
| `image` | string | Source image |
| `cy`, `cx` | float | Centroid in ROI coordinates (row, col) |
| `area` | float | Connected-component area (pixels) |
| `ecc` | float | Eccentricity (0 = circle, →1 = line) |
| `major`, `minor` | float | Major and minor axis lengths (pixels) |
| `circ` | float | Circularity = 4π·area/perimeter² |
| `mean_int` | float | Mean intensity inside the blob |
| `max_int` | float | Peak intensity in the 48×48 patch |
| `cluster` | int | KMeans cluster label (0–5) |
| `is_track` | bool | `True` = counted as real track; `False` = artifact cluster |

**Use:** inspect individual detections, re-filter with different `ARTIFACT_CLUSTERS`, correlate morphology with exposure conditions.

---

#### `final_counts.csv`

Per-image summary after unsupervised cleaning.

| Column | Type | Description |
|--------|------|-------------|
| `image` | string | JPEG filename |
| `imagej_count` | int | Macro baseline |
| `macro_repro` | int | Python macro reproduction |
| `improved` | int | All improved candidates |
| `repro_err_%` | float | Macro reproduction error vs ImageJ |
| `extra_vs_macro` | int | Extra candidates vs macro |
| `extra_%` | float | Extra candidates (%) |
| `nn_clean_count` | int | **Final NN-cleaned track count** |
| `cluster_0`, `cluster_2`, … | int | Per-image count in each real-track cluster |

Only clusters marked `is_track = True` contribute to `nn_clean_count` and the per-cluster columns.

**Use:** primary result table for comparing radon exposure across detector sheets.

---

#### `embeddings.npy`

NumPy array, shape `(N, 32)` where N = number of candidates.

- Each row is the **32-D autoencoder latent vector** for one candidate patch.
- Dtype: `float32`.
- Load with: `emb = np.load("embeddings.npy")`

**Use:** custom clustering, dimensionality reduction, or as features for a supervised classifier later.

---

#### `step4_scores.csv` *(created after ground-truth annotation)*

Precision/recall table for the evaluation crop.

| Column | Description |
|--------|-------------|
| Index | Method name (`macro`, `improved`, `NN_clean`, `cellpose`) |
| `n_det` | Number of detections in the crop |
| `TP`, `FP`, `FN` | True/false positives and false negatives |
| `precision`, `recall`, `f1` | Standard metrics (0–1) |

**Use:** objectively pick the best method and tune `ARTIFACT_CLUSTERS` / `MATCH_RADIUS`.

---

#### `gt_<image>_<x0>_<y0>_<size>.csv` *(created by user annotation)*

Ground-truth track centres for one evaluation crop.

| Column | Description |
|--------|-------------|
| `x_crop`, `y_crop` | Click position in crop coordinates |
| `x_abs`, `y_abs` | Same point in full ROI coordinates |

Example filename: `gt_LBS255611_400_2800_700.csv`

**Use:** cached hand counts; reload to refine or re-score without re-clicking.

---

### 5.2 PNG figures

#### `step1_counts.png`

Grouped bar chart: per image, three bars —

- blue = ImageJ macro count
- orange = macro reproduced in Python
- green = improved detector count

Plus a scatter plot of detector count vs macro count.

---

#### `step1_overlay.png`

Side-by-side on a dense 700×700 sub-crop:

- Left: contrast-stretched raw crop
- Right: same crop with detection markers
  - **Green** = macro would keep (round, size OK)
  - **Red** = macro would drop (elongated, merged, or out of size range)

**Use:** visual proof of what the circularity filter removes.

---

#### `step2_cellpose_subcrop.png`

Side-by-side:

- Left: sub-crop (contrast-stretched)
- Right: Cellpose instance masks overlaid (coloured regions)

**Use:** assess whether Cellpose splits touching tracks better than the macro.

---

#### `cluster_0.png` … `cluster_5.png`

Montage of ~24 random 48×48 candidate patches per cluster, with title showing:

- cluster id and population `n`
- mean area, eccentricity, circularity, max intensity

**Use:** the primary tool for deciding which clusters are real tracks vs artifacts. Edit `ARTIFACT_CLUSTERS` based on these.

---

#### `step3_tsne.png`

2D t-SNE projection of candidate embeddings, points coloured by cluster.

**Use:** see whether clusters are well-separated in feature space; guide choice of `n_clusters`.

---

#### `final_counts.png`

Grouped bar chart per image:

- blue = ImageJ macro
- orange = improved (all candidates)
- green = NN-cleaned (artifacts removed)

**Use:** quick visual comparison of the three counting strategies.

---

#### `step4_tp_fp_fn.png` *(created after ground-truth annotation)*

One panel per method (`macro`, `improved`, `NN_clean`) on the evaluation crop:

- **Green** = true positive
- **Red** = false positive
- **Yellow square** = false negative

Title shows precision, recall, F1, and TP/FP/FN counts.

**Use:** diagnose *how* each method fails (misses oblique tracks? counts dust?).

---

## 6. Typical workflow

1. Open `algorithms/radon_track_nn.ipynb` with kernel **"Python (pytmetalbeta · MPS)"**.
2. Run all cells (Steps 0–3 produce outputs automatically).
3. Inspect `cluster_*.png` montages; adjust `ARTIFACT_CLUSTERS` if needed; re-run Step 3.8.
4. Read `final_counts.csv` for per-image radon track counts.
5. (Recommended) Run Step 4: click tracks on the evaluation crop, save GT, score all methods.
6. Use `step4_scores.csv` and `step4_tp_fp_fn.png` to choose the operating point.

---

## 7. Interpretation notes and limitations

**Macro reproduction vs ImageJ (~12% offset):** small but systematic; caused by differences in perimeter measurement, 8-connectivity handling, and floating-point vs integer area. The trend (which images have more/fewer tracks) is preserved.

**Improved > macro (+33%):** largely real elongated and merged tracks the circularity gate rejects. Some fraction may be scratches split into multiple detections — check the overlay.

**NN-cleaned < macro:** artifact clusters (1, 4) remove thousands of faint specks. The NN count is **not directly comparable** to the macro without Step 4 validation; it is a different definition of "track".

**Cellpose on sub-crop:** under-segments faint CR-39 tracks out of the box; useful as a baseline to tune, not as a drop-in replacement without parameter adjustment.

**Unsupervised clustering:** requires one human pass on cluster montages. Cluster semantics may shift if you change `det_fixed_thresh`, `n_clusters`, or `ARTIFACT_CLUSTERS`.

**No ground truth yet:** `step4_scores.csv` and `step4_tp_fp_fn.png` do not exist until you annotate interactively. The notebook runs headlessly without errors but skips scoring.

---

## 8. File inventory summary

| File | Step | Format | Created when |
|------|------|--------|--------------|
| `step1_counts.csv` | 1 | CSV | Always |
| `step1_counts.png` | 1 | PNG | Always |
| `step1_overlay.png` | 1 | PNG | Always |
| `step2_cellpose_subcrop.png` | 2 | PNG | Always |
| `embeddings.npy` | 3 | NumPy | Always |
| `candidates_labeled.csv` | 3 | CSV | Always |
| `cluster_0.png` … `cluster_5.png` | 3 | PNG | Always |
| `step3_tsne.png` | 3 | PNG | Always |
| `final_counts.csv` | 3 | CSV | Always |
| `final_counts.png` | 3 | PNG | Always |
| `gt_*.csv` | 4 | CSV | After user annotation |
| `step4_scores.csv` | 4 | CSV | After GT saved |
| `step4_tp_fp_fn.png` | 4 | PNG | After GT saved |
