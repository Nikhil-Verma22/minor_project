# SGG Benchmark Evaluation Sets — Technical Manifest & AI Context Digest

> **Purpose for AI Systems**: This document provides an information-dense, structured technical digest of the `evaluation_sets/` directory. Rather than duplicating raw data, it synthesizes the architecture, dataset ontologies, normalization frameworks, audit verifications, schemas, and non-readable binary assets so an AI agent can fully understand and reason about the benchmark suite without overflowing its context window.

---

## 1. Executive Summary & Benchmark Overview

The `evaluation_sets` folder is a standardized, audited evaluation testbed for **Scene Graph Generation (SGG)** models (evaluating architectures such as **REACT**, **REACT++**, **FCSGG**, and **Fast Contextual SGG**).

### Core Highlights:
- **Multi-Dataset Coverage**: Standardizes four major SGG datasets: **VG150**, **PSG**, **IndoorVG**, and **Open Images V6**.
- **Deterministic Sampling**: Exactly **200 images** per dataset (**800 images total**), sampled deterministically from official test splits to enable rapid yet statistically representative cross-dataset evaluation.
- **Cross-Ontology Normalization**: Solves the vocabulary mismatch between datasets by providing **strict (1-to-1)**, **extended (semantic generalizations)**, and **tri-dataset (3-way)** alignment views.
- **Integrity Verification**: 100% verified by an automated 12-check pre-evaluation audit and validated via an end-to-end ONNX smoke test.

---

## 2. Directory Tree & File Manifest

```text
evaluation_sets/
├── images/                                  [800 JPG images, 137.87 MB total — Non-readable binary]
│   ├── IndoorVG/                            [200 JPG images, 22.47 MB; avg 465×393 px]
│   ├── OpenImages/                          [200 JPG images, 56.72 MB; avg 936×811 px]
│   ├── PSG/                                 [200 JPG images, 33.13 MB; avg 586×468 px]
│   ├── VG150/                               [200 JPG images, 25.55 MB; avg 465×395 px]
│   └── .DS_Store                            [6,148 bytes, macOS system cache binary]
├── normalized/                              [13 cross-dataset aligned annotations & summary, 9.61 MB]
│   ├── IndoorVG_200_extended.coco.json      [1,028 KB; relaxed alignment with semantic generalization]
│   ├── IndoorVG_200_strict.coco.json        [1,028 KB; strict 1-to-1 common predicate vocabulary]
│   ├── IndoorVG_200_tri_dataset.coco.json   [1,145 KB; 3-way alignment across VG, PSG, IndoorVG]
│   ├── OpenImages_200_extended.coco.json    [437 KB; extended predicate alignment]
│   ├── OpenImages_200_strict.coco.json      [436 KB; strict predicate alignment]
│   ├── OpenImages_200_tri_dataset.coco.json [457 KB; 3-way alignment]
│   ├── PSG_200_extended.coco.json           [899 KB; extended predicate alignment]
│   ├── PSG_200_strict.coco.json             [883 KB; strict predicate alignment]
│   ├── PSG_200_tri_dataset.coco.json        [1,042 KB; 3-way alignment]
│   ├── VG150_200_extended.coco.json         [738 KB; extended predicate alignment]
│   ├── VG150_200_strict.coco.json           [738 KB; strict predicate alignment]
│   ├── VG150_200_tri_dataset.coco.json      [814 KB; 3-way alignment]
│   └── vg_psg_normalization_summary.json    [1,885 bytes; retention & drop rate statistics]
├── smoke_test_output/                       [Verification & smoke-test execution outputs]
│   ├── eval.log                             [4,158 bytes; inference runtime trace log]
│   ├── eval_results_top_100.json            [1,060 bytes; Recall@K & Mean Recall metrics]
│   └── eval_summary.json                    [2,212 bytes; latency (153ms), hardware, model config]
├── .DS_Store                                [14,340 bytes, macOS system cache binary]
├── IndoorVG_200.txt                         [1,200 bytes; 200 selected image IDs for IndoorVG]
├── IndoorVG_200_annotations.coco.json       [966 KB; native COCO annotations (84 cats, 37 predicates)]
├── IndoorVG_200_metadata.json               [49,542 bytes; test split selection parameters & image metadata]
├── OpenImages_200.txt                       [3,400 bytes; 200 selected image IDs for OpenImages V6]
├── OpenImages_200_annotations.coco.json     [561 KB; native COCO annotations (200 images)]
├── OpenImages_200_metadata.json             [54,071 bytes; selection parameters & image metadata]
├── PSG_200.txt                              [1,551 bytes; 200 selected image IDs for PSG]
├── PSG_200_annotations.coco.json           [919 KB; native COCO annotations (133 cats, 56 predicates)]
├── PSG_200_metadata.json                   [3,347 bytes; selection metadata & commit hashes]
├── PSG_download_metadata.json              [382 bytes; image verification report (100% verified)]
├── VG150_200.txt                            [1,600 bytes; 200 selected image IDs for VG150]
├── VG150_200_annotations.coco.json         [707 KB; native COCO annotations (150 cats, 50 predicates)]
├── VG150_200_metadata.json                 [3,412 bytes; selection metadata & commit hashes]
├── VG150_download_metadata.json            [342 bytes; image verification report (100% verified)]
├── benchmark_validation_summary.json        [4,285 bytes; summary GT objects & relations metrics]
├── dataset_preparation_report.md           [13,461 bytes; dataset construction documentation]
├── evaluation_harness_smoke_test.md        [10,556 bytes; harness smoke test report]
├── evaluation_ontology.json                [55,881 bytes; cross-dataset unified ontology mapping]
├── evaluation_readiness_checklist.json     [8,188 bytes; readiness protocol sign-off checklist]
├── evaluation_readiness_report.md          [13,332 bytes; readiness audit resolution report]
├── pre_evaluation_audit.json               [18,673 bytes; automated audit report structured data]
├── pre_evaluation_audit.md                 [20,088 bytes; automated audit documentation report]
├── pre_evaluation_audit_checks.csv         [10,235 bytes; check-by-check audit verification table]
├── relationship_alignment_audit.json       [40,812 bytes; relationship alignment audit data]
├── relationship_alignment_audit.md         [20,677 bytes; relationship vocabulary alignment audit report]
├── relationship_alignment_candidates.json  [42,313 bytes; candidate predicate alignment data]
├── relationship_alignment_candidates.md    [20,665 bytes; candidate predicate alignment report]
├── relationship_alignment_final.json       [57,396 bytes; frozen final evaluation ontology data]
├── relationship_alignment_final.md         [25,674 bytes; frozen final evaluation ontology report]
├── relationship_alignment_reviewed.json    [68,483 bytes; second-pass reviewed alignment data]
├── relationship_alignment_reviewed.md      [26,708 bytes; second-pass reviewed alignment report]
├── relationship_vocabularies.json          [12,936 bytes; authoritative raw original predicate vocabularies]
└── relationship_vocabularies.md            [7,812 bytes; authoritative predicate vocabularies report]
```

---

## 3. Dataset Breakdown & Quantitative Metrics

| Dataset | Split | Images | GT Objects | GT Relationships | Object Classes | Predicate Classes | Avg Image Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VG150** | Test | 200 | 2,320 | 1,083 | 150 | 50 | 465 × 395 px |
| **PSG** | Test | 200 | 3,019 | 1,385 | 133 | 56 | 586 × 468 px |
| **IndoorVG** | Test | 200 | 3,118 | 1,672 | 84 | 37 | 465 × 393 px |
| **OpenImages V6** | Test | 200 | 1,486 | 866 | 601 | 30 | 936 × 811 px |
| **TOTAL / POOL** | — | **800** | **9,943** | **5,006** | — | — | — |

---

## 4. Non-Readable Files Inventory (Images & Binaries)

The directory contains **800 JPEG image files** and **2 macOS system binary files**.

### 4.1 Evaluation Images (`evaluation_sets/images/`)
All images are standard 3-channel RGB JPEGs verified for integrity, correct dimensions matching annotation headers, and zero corruption.
- **`images/IndoorVG/`**: 200 JPG files (22.47 MB). Width: 180–500 px (avg 465 px), Height: 178–500 px (avg 393 px).
- **`images/OpenImages/`**: 200 JPG files (56.72 MB). Width: 365–1024 px (avg 936 px), Height: 447–1024 px (avg 811 px).
- **`images/PSG/`**: 200 JPG files (33.13 MB). Width: 320–640 px (avg 586 px), Height: 164–640 px (avg 468 px).
- **`images/VG150/`**: 200 JPG files (25.55 MB). Width: 281–500 px (avg 465 px), Height: 271–500 px (avg 395 px).
- **Verification Status**: `PSG_download_metadata.json` and `VG150_download_metadata.json` confirm `verification_status: "SUCCESS"` with 0 missing IDs, 0 failed downloads, and 0 dimension mismatches.

### 4.2 System Binary Files
- **`evaluation_sets/.DS_Store`** (14,340 bytes) and **`evaluation_sets/images/.DS_Store`** (6,148 bytes): macOS Finder desktop metadata cache files. Safe to ignore during automated evaluation.

---

## 5. COCO Annotation Schema & Triplet Structure

All 16 annotation files (4 root native files and 12 normalized files) conform strictly to an extended COCO Scene Graph format.

### Key Top-Level Fields:
```json
{
  "images": [
    {"id": 73306, "file_name": "2346196.jpg", "width": 500, "height": 375}
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 73306,
      "category_id": 61,
      "bbox": [391.0, 99.0, 85.0, 74.0],
      "area": 6290.0,
      "iscrowd": 0,
      "segmentation": []
    }
  ],
  "rel_annotations": [
    {
      "id": 0,
      "image_id": 73306,
      "subject_id": 0,
      "object_id": 1,
      "predicate_id": 25
    }
  ],
  "categories": [
    {"id": 1, "name": "bag", "supercategory": "none"}
  ],
  "rel_categories": [
    {"id": 1, "name": "above", "supercategory": "none"}
  ]
}
```

### Critical Semantic Linking Rules:
- `subject_id` and `object_id` in `rel_annotations` link directly to the annotation entry's `id` in the `annotations` list (the entity bounding boxes).
- `predicate_id` references the `id` in `rel_categories`.
- Bounding boxes are formatted as standard COCO `[x_min, y_min, width, height]` in non-normalized absolute image pixels.

---

## 6. Cross-Dataset Normalization Framework (`normalized/`)

Direct cross-dataset evaluation fails when models trained on one predicate vocabulary are tested against ground truth annotated under a different vocabulary. The `normalized/` folder provides systematic predicate mappings:

### 1. Strict 4-Way (`strict`)
- **Philosophy**: Retains only relationships where an exact 1-to-1 semantic equivalence exists across all four datasets.
- **Behavior**: Non-aligned relationships are excluded from ground truth, penalizing neither precision nor recall for vocabulary divergence.
- **Retention Rates**:
  - **VG150**: Retains 604 / 1,083 relations (55.77% retention; 479 dropped).
  - **PSG**: Retains 520 / 1,385 relations (37.55% retention; 865 dropped).

### 2. Extended 4-Way (`extended`)
- **Philosophy**: Adds high-confidence semantic generalizations (hypernym/hyponym mapping) to the strict set (e.g., mapping `sitting on` / `standing on` to `on` where appropriate).
- **Retention Rates**:
  - **VG150**: Retains 605 / 1,083 relations (55.86% retention).
  - **PSG**: Retains 575 / 1,385 relations (41.52% retention).

### 3. Tri-Dataset View (`tri_dataset`)
- **Philosophy**: 3-way alignment optimized across Visual Genome, PSG, and IndoorVG / OpenImages.
- **Retention Rates**:
  - **VG150**: Retains 873 / 1,083 relations (80.61% retention; only 210 dropped).
  - **PSG**: Retains 1,050 / 1,385 relations (75.81% retention; 335 dropped).

---

## 7. Technical Reports & Audit Documentation Summary

The repository includes 9 Markdown reports and 3 verification logs that establish the scientific and engineering foundation of the benchmark:

### 1. Dataset Preparation & Readiness
- **`dataset_preparation_report.md`**: Explains the end-to-end dataset generation pipeline, selection criteria, bounding box cleaning, and integration into the benchmark evaluation harness.
- **`evaluation_readiness_report.md` & `evaluation_readiness_checklist.json`**: Official evaluation protocol sign-off confirming commit hashes, environment flags, baseline checkpoints, and evaluation command lines.

### 2. Pre-Evaluation Quality Audit
- **`pre_evaluation_audit.md`, `pre_evaluation_audit.json`, `pre_evaluation_audit_checks.csv`**: Automated audit executing 12 integrity checks with a 100% pass rate:
  1. Image file existence on disk (200/200 for all 4 datasets).
  2. Image header width/height matching COCO metadata.
  3. Bounding box validity: `0 <= x < width` and `0 <= y < height`.
  4. Positive box dimensions: `width > 0` and `height > 0`.
  5. Bounding box area consistency: `area == width * height`.
  6. Subject ID exists in `annotations`.
  7. Object ID exists in `annotations`.
  8. Predicate ID exists in `rel_categories`.
  9. Non-empty categories and rel_categories catalogs.
  10. Zero duplicate image IDs.
  11. Deterministic image ID ordering.
  12. Checkpoint availability and loading test.

### 3. Relationship Alignment & Ontology Evolution
- **`relationship_vocabularies.md` & `.json`**: Authoritative original unmapped predicate lists (VG150: 50, PSG: 56, IndoorVG: 37, OpenImages: 30).
- **`relationship_alignment_candidates.md` & `.json`**: Phase 1 conservative semantic alignment matrix.
- **`relationship_alignment_reviewed.md` & `.json`**: Phase 2 multi-expert review expanding coverage for complex spatial and interaction predicates (`holding`, `carrying`, `wearing`, `attached to`).
- **`relationship_alignment_final.md` & `relationship_alignment_final.json`**: Frozen evaluation ontology formally locking category indices and mapping tables for all benchmark runs.
- **`relationship_alignment_audit.md` & `.json`**: Audit report analyzing coverage trade-offs, ambiguity resolutions, and edge cases.

### 4. Harness Smoke-Test Verification
- **`evaluation_harness_smoke_test.md`**: Technical report verifying the PyTorch/ONNX evaluation harness.
- **`smoke_test_output/eval.log`**: Runtime trace evaluating checkpoint `checkpoints/PSG/react++_yolo12m` on PSG strict evaluation split.
- **`smoke_test_output/eval_summary.json`**: Execution performance: **153.15 ms** mean forward latency per image (approx. 6.5 FPS), zero runtime exceptions, valid SGDet task outputs.
- **`smoke_test_output/eval_results_top_100.json`**: Initial metric outputs (Recall@20, Recall@50, Recall@100, Mean Recall, F1 score).

---

## 8. Summary for AI Systems & Automated Evaluators

When using `evaluation_sets/` for model inference or benchmark evaluation:
1. **Input Images**: Load from `evaluation_sets/images/<Dataset>/` using the image filenames listed in `<Dataset>_200_annotations.coco.json`.
2. **Ground Truth Selection**:
   - For **native single-dataset evaluation**: Use root annotations `<Dataset>_200_annotations.coco.json`.
   - For **cross-dataset zero-shot / transfer evaluation**: Use `normalized/<Dataset>_200_strict.coco.json` (for high-precision strict evaluation) or `normalized/<Dataset>_200_extended.coco.json` (for higher recall).
3. **Evaluation Metric Protocol**: Standard SGG evaluation computes **R@20, R@50, R@100** and **mR@20, mR@50, mR@100** under three standard tasks:
   - **PredCls** (Predicate Classification): Ground truth boxes and object categories provided.
   - **SGCls** (Scene Graph Classification): Ground truth boxes provided; object categories and predicates predicted.
   - **SGDet** (Scene Graph Detection): Object boxes, categories, and predicates all predicted from raw images.
