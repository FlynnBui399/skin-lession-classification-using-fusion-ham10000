# Dataset — HAM10000

## Source

**Name:** HAM10000 — *Human Against Machine with 10000 training images*

**Primary reference:** Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* **Scientific Data**, 5, 180161. <https://doi.org/10.1038/sdata.2018.161>

**Canonical download (Harvard Dataverse):** DOI `10.7910/DVN/DBW86T` — <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T>

**Kaggle mirror used in this project:** <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>

**License:** CC BY-NC 4.0 (non-commercial research use).

## Description

HAM10000 is a dermatoscopic image dataset collected over ~20 years at two clinical sites (Department of Dermatology, Medical University of Vienna, Austria; skin cancer practice of Cliff Rosendahl, Queensland, Australia). It was released as a benchmark for multi-class pigmented skin lesion classification.

- **Observations:** 10,015 dermatoscopic images.
- **Image resolution:** 600 × 450 px (RGB, JPEG).
- **Target variable `dx`:** 7 diagnostic classes.
- **Ground-truth methods (`dx_type`):** histopathology (`histo`), follow-up examination (`follow_up`), expert consensus (`consensus`), in-vivo confocal microscopy (`confocal`).

### Class distribution

| Code | Diagnosis | Count | Share |
|------|-----------|------:|------:|
| `nv`    | Melanocytic nevi                              | 6,705 | 66.9% |
| `mel`   | Melanoma                                      | 1,113 | 11.1% |
| `bkl`   | Benign keratosis-like lesions                 | 1,099 | 11.0% |
| `bcc`   | Basal cell carcinoma                          |   514 |  5.1% |
| `akiec` | Actinic keratoses / intraepithelial carcinoma |   327 |  3.3% |
| `vasc`  | Vascular lesions                              |   142 |  1.4% |
| `df`    | Dermatofibroma                                |   115 |  1.1% |

The dataset is **severely imbalanced**; macro-averaged metrics and per-class recall (especially for `mel`) are therefore more informative than raw accuracy.

### Metadata variables (`HAM10000_metadata.csv`)

| Variable       | Type        | Description |
|----------------|-------------|-------------|
| `lesion_id`    | categorical | Identifier of the physical lesion (one lesion can have multiple images). |
| `image_id`     | categorical | Unique image identifier (matches the `.jpg` filename). |
| `dx`           | categorical | Diagnosis — target variable (7 classes). |
| `dx_type`      | categorical | Method used to establish the ground-truth label. |
| `age`          | continuous  | Patient age in years (some missing). |
| `sex`          | categorical | `male`, `female`, `unknown`. |
| `localization` | categorical | Body site of the lesion (15 levels). |

Derived image features (50 handcrafted descriptors — RGB / HSV statistics, brightness, contrast, asymmetry, border irregularity, LBP and GLCM texture) are produced by the Report notebook and cached in `results/image_features.csv`.

## Expected folder layout

The notebooks in this repository expect the following layout relative to the project root:

```
data/
├── README.md                       (this file)
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/         (5,000 JPEG images)
├── HAM10000_images_part_2/         (5,015 JPEG images)
└── HAM10000_images/                (optional: merged folder used by the notebooks)
```

> If you keep the two original part folders, the Report notebook will build an `image_id → path` index covering both. A merged `HAM10000_images/` folder is **not required**.

## Download instructions

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/ (Linux/macOS) or %USERPROFILE%\.kaggle\ (Windows)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ --unzip
```

### Option B — Harvard Dataverse (official)

1. Open <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T>.
2. Download `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`, and `HAM10000_metadata.tab`.
3. Unzip the two archives into `data/` and rename `HAM10000_metadata.tab` to `HAM10000_metadata.csv`.

## Data-quality notes

- **Labels are not all histologically verified.** Only `dx_type == histo` rows (~53% of records) have biopsy-level ground truth; the rest rely on follow-up, consensus, or confocal imaging. This limits the ceiling of supervised performance and should be discussed when interpreting results.
- **Patient leakage.** Multiple images can share the same `lesion_id`. Stratified splits on `image_id` can leak information across train/test; splitting on `lesion_id` is the stricter protocol.
- **Missing `age`.** A small fraction (~0.6%) of records have missing age; the Report imputes with the median.
- **Source heterogeneity.** Images come from multiple clinics and acquisition devices, which contributes to domain shift and was one motivation for publishing the dataset in the first place.

## Citation

If you use this dataset, please cite the original paper:

```bibtex
@article{tschandl2018ham10000,
  title   = {The {HAM10000} dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author  = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180161},
  year    = {2018},
  doi     = {10.1038/sdata.2018.161}
}
```
