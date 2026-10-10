# Fast Contextual SGG (C-SGG & CV-SGG) Benchmark Evaluation Report

**Benchmark Project**: Object Retrieval in Dynamic Workspace Environments  
**Workspace Root**: `d:/Minor`  
**Model Family**: Fast Contextual Scene Graph Generation with Unbiased Context Augmentation  
**Repository**: `model_repos/fast-context-scene-graph-generation/`  
**Upstream**: [https://github.com/moshuilanting/fast-context-scene-graph-generation](https://github.com/moshuilanting/fast-context-scene-graph-generation)  
**Hardware Platform**: Intel(R) Core(TM) Ultra 5 125H (18 logical threads), Windows 11  
**Software Environment**: Python 3.13.7, PyTorch 2.8.0+cpu, torchvision 0.23.0, OpenCV 4.11.0, pycocotools 2.0.8  
**Evaluation Protocol**: 200-image frozen subset (`evaluation_sets/`), dual empty-ground-truth policy, multi-ontology views  
**Date**: October 10, 2026  

---

## Executive Summary

This report establishes the empirical evaluation of **Fast Contextual Scene Graph Generation** within the standardized benchmark protocol defined in Prompt 1 (REACT), Prompt 2 (REACT++), and Prompt 3 (FCSGG).

Fast Contextual SGG introduces a context-first paradigm designed to overcome the quadratic computational bottleneck of dense visual relation extraction. Two variants were evaluated separately:
1. **C-SGG (Contextual-only Scene Graph Generation)**: Generates relationship predictions using only semantic word embeddings (GloVe 50d) and 2D bounding box spatial configurations via a lightweight **Context Knowledge Network (CKN)**.
2. **CV-SGG (Contextual + Visual Verification)**: Uses CKN as an initial contextual filter to prune unpromising pairs by 94.2% (from 2,580 candidates down to ~149 candidate pairs), followed by a **Vision Differentiation Network (VDN)** that verifies candidate relations against 5-channel image crops with subject/object binary masks.

Both variants were evaluated on the **200-image frozen Visual Genome (VG150)** benchmark across all four ontology views (`strict`, `extended`, `tri_dataset`, and `native`). Pre-evaluation inspection determined that **OpenPSG**, **Open Images V6**, and **IndoorVG** are **BLOCKED** due to offline precomputed file dependencies and dataset contamination.

```mermaid
flowchart TD
    Img[Raw Image 640x640] --> Det[YOLOv5l Object Detector]
    Det --> Bboxes[Detected Bounding Boxes & Categories N <= 80]
    Bboxes --> Pairs[Generate N x N-1 Directed Candidate Pairs ~2,580 pairs]
    Pairs --> CKN[Context Knowledge Network CKN 190-dim MLP]
    CKN --> CSGG[C-SGG Scene Graph Output Latency: 4.96 ms | FPS: 2.85]
    CKN --> Filter[Candidate Filtering Top K = 150 Pairs 94.2% Pruned]
    Filter --> Crops[5-Channel Image & Binary Mask Crops 224x224]
    Crops --> VDN[Vision Differentiation Network ResNet + Fusion]
    VDN --> CVSGG[CV-SGG Scene Graph Output Latency: 4,516 ms | FPS: 0.22]
```

---

## 1. Pretrained Weights, Dependencies & Audit

### 1.1 Model Checkpoint Inventory
- **VG CKN (`CKN+EPBS.pt`)**: 828 KB, 3-layer MLP (`190 -> 256 -> 512 -> 51`), 331,827 parameters. Committed in repository and loaded cleanly on PyTorch CPU.
- **VG VDN (`CKN+VDN+EPBS.pt`)**: 19.2 MB, custom ResNet Bottleneck backbone `[2, 2, 2, 2, 2]` with visual-semantic fusion head, 4,785,967 parameters. Committed in repository and loaded cleanly on PyTorch CPU.
- **VG Object Detector (`yolol_object_test_28.pt`)**: 359 MB (376,488,983 bytes), YOLOv5l trained on Visual Genome 150 object classes (`nc=151`). Downloaded from official Google Drive link to `checkpoints/FastContextualSGG/yolol_object_test_28.pt`.
- **Word Embeddings**: Standard 50-dimensional GloVe vectors for the 151 Visual Genome classes were streamed and cached locally in `.cache/FastContextualSGG/vg_word_vec_list.pt` (30 KB), fully bypassing deprecated `torchtext`.

### 1.2 Dataset Compatibility Matrix

| Dataset | Split | C-SGG Status | CV-SGG Status | Audit Findings & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Visual Genome (VG150)** | 200 images | **SUPPORTED** | **SUPPORTED** | Full native 150 object & 50 predicate support. YOLOv5l detector, CKN, and VDN all active. |
| **OpenPSG (PSG)** | 200 images | **BLOCKED** | **BLOCKED** | No online detector in repo; strictly depends on offline precomputed Panoptic SegFormer PVTv2 masks (`SegFromer_PVTV/`). |
| **Open Images V6 (OIDv6)**| 200 images | **BLOCKED** | **BLOCKED** | Implementation strictly requires offline precomputed Faster R-CNN detection pickle files (`bbox_save.bin`). |
| **IndoorVG** | 200 images | **BLOCKED** | **BLOCKED** | 100% contaminated with VG150 training split (`LEAK_INDOOR_VGTRAIN`). No method-specific checkpoints exist. |

---

## 2. Evaluation Results Across Ontology Views

All evaluations were executed on the frozen 200-image Visual Genome dataset under the dual empty-ground-truth policy:
- **Eligible Images**: Evaluated only on images containing $\ge 1$ ground-truth relation in the specified ontology view.
- **All 200 Images Sensitivity**: Full-set sensitivity metric setting empty-ground-truth images to 0.0 recall.

### 2.1 C-SGG (Contextual-Only Relationship Prediction)

| View | Eligible / Total Imgs | R@20 | R@50 | R@100 | R@100 (All 200) | mR@20 | mR@50 | mR@100 | Det mAP@50 | F1@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict** (6 preds) | 141 / 200 | 0.80% | 1.09% | 3.58% | 2.52% | 0.20% | 0.26% | 1.15% | 0.3419 | 0.4137 |
| **Extended** (8 preds) | 142 / 200 | 0.70% | 0.80% | 0.94% | 0.67% | 0.12% | 0.17% | 0.23% | 0.3419 | 0.4137 |
| **Tri-Dataset** (20 preds)| 160 / 200 | 0.31% | 0.38% | 0.51% | 0.41% | 0.05% | 0.07% | 0.09% | 0.3419 | 0.4137 |
| **Native** (50 preds) | 174 / 200 | **10.35%**| **18.78%**| **24.85%**| **21.62%** | **9.53%**| **13.57%**| **15.33%**| **0.3419** | **0.4137** |

### 2.2 CV-SGG (Contextual Filtering + Visual Verification)

| View | Eligible / Total Imgs | R@20 | R@50 | R@100 | R@100 (All 200) | mR@20 | mR@50 | mR@100 | Det mAP@50 | F1@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict** (6 preds) | 141 / 200 | 0.71% | 1.13% | 2.03% | 1.43% | 0.14% | 0.34% | 0.64% | 0.3419 | 0.4137 |
| **Extended** (8 preds) | 142 / 200 | 0.70% | 0.70% | 0.70% | 0.50% | 0.12% | 0.12% | 0.12% | 0.3419 | 0.4137 |
| **Tri-Dataset** (20 preds)| 160 / 200 | 0.31% | 0.31% | 0.31% | 0.25% | 0.05% | 0.05% | 0.05% | 0.3419 | 0.4137 |
| **Native** (50 preds) | 174 / 200 | **8.94%** | **16.43%**| **24.12%**| **20.98%** | **8.50%**| **11.73%**| **13.84%**| **0.3419** | **0.4137** |

---

## 3. Computational Complexity & Latency Breakdown

The design of Fast Contextual SGG allows explicit separation of detector, contextual prediction, and visual verification stages:

| Metric / Stage | C-SGG (Context-Only) | CV-SGG (Context + Visual) | Delta / Ratio |
| :--- | :---: | :---: | :---: |
| **Object Detector Latency** | 340.43 ms | 439.22 ms | +29.0% (due to RAM load) |
| **Contextual Reasoning (CKN)** | **4.96 ms** | **6.18 ms** | Extremely lightweight |
| **Visual Verification (VDN)** | **0.00 ms** | **4,064.74 ms** | Bottleneck stage on CPU |
| **Total End-to-End Latency** | **350.34 ms** | **4,516.57 ms** | **12.9x slower** |
| **Throughput (FPS)** | **2.85 FPS** | **0.22 FPS** | **12.9x reduction** |
| **Peak RAM RSS** | 935.3 MB | 2,271.9 MB | +1,336.6 MB |
| **Candidate Pairs Evaluated (CKN)** | 2,580.0 pairs | 2,580.0 pairs | All $N(N-1)$ pairs |
| **Candidate Pairs Verified (VDN)** | 0.0 pairs | **148.9 pairs** | **94.2% candidates filtered** |

### Key Architectural Takeaways:
1. **Context Knowledge Network (CKN) is near-zero overhead**: Processing an average of 2,580 directed pairs per image through CKN requires only **4.96 ms** on CPU.
2. **Contextual filtering prunes 94.2% of pairs**: The CKN filter reduces the pair set from 2,580 candidates down to ~149 pairs for visual verification.
3. **CPU Bottleneck in VDN**: On CPU, extracting 149 image patches and evaluating them through ResNet takes ~4.06 seconds per image. In GPU/TensorRT environments (as demonstrated in the original paper), this stage benefits heavily from CUDA parallelization.

---

## 4. Multi-Model Benchmark Comparison (Minor Workspace)

Comparing Fast Contextual SGG against prior benchmarks evaluated in the Minor workspace on the frozen Visual Genome VG150 dataset:

| Model / Variant | Backbone / Detector | Native R@50 | Native R@100 | Native mR@100 | Det mAP@50 | E2E Latency (ms) | Throughput (FPS) | Peak RAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C-SGG** | YOLOv5l + CKN (MLP) | **18.78%** | **24.85%** | **15.33%** | 0.3419 | **350.3 ms** | **2.85** | **935.3** |
| **CV-SGG** | YOLOv5l + CKN + VDN | **16.43%** | **24.12%** | **13.84%** | 0.3419 | **4,516.6 ms** | **0.22** | 2,271.9 |
| **FCSGG** | HRNetW32-1S (CenterNet)| 20.37% | 23.36% | 11.08% | 0.2818 | 1,277.1 ms | 0.78 | 3,160.6 |
| **REACT** | YOLOv8m (PyTorch) | 8.87% | 12.35% | 7.91% | 0.3204 | 485.6 ms | 2.06 | 1,420.0 |

### Insights:
- **C-SGG achieves the highest Native Mean Recall (15.33%)** among all evaluated models in the benchmark, while also delivering the fastest inference speed (350.3 ms E2E, 2.85 FPS on CPU).
- **YOLOv5l detector strength**: Achieved mAP@50 of 0.3419, surpassing CenterNet/HRNetW32 (0.2818) and YOLOv8m (0.3204).
- **C-SGG vs CV-SGG Tradeoff**: In the presence of object bounding box localization noise from online detection, CKN's contextual priors alone generalize better to candidate recall than VDN's visual verification, which penalizes imperfect bounding box crops.

---

## 5. Per-Predicate Recall Analysis & Failure Modes

### 5.1 Top-Performing Predicates (Native View R@100)
- `parked on`: **100.0%** (GT: 2) — Strong spatial and semantic correlation between vehicles and parking surfaces.
- `riding`: **76.7%** (C-SGG) / **50.0%** (CV-SGG) (GT: 8) — Clear person-vehicle physical configuration.
- `at`: **50.0%** (GT: 2) — Positional context.
- `has`: **49.5%** (C-SGG) / **38.8%** (CV-SGG) (GT: 149) — Part-whole context (e.g. `car has wheel`, `person has arm`).
- `sitting on`: **40.5%** (GT: 12) — Canonical posture and furniture interaction.
- `wearing`: **40.8%** (GT: 113) — Person-clothing contextual association.
- `of`: **32.5%** (C-SGG) / **34.7%** (CV-SGG) (GT: 77) — Semantic possession.

### 5.2 Failure Cases and Bottlenecks
1. **Rare & Long-Tail Predicates**: Predicates such as `across`, `attached to`, `belonging to`, `covering`, `eating`, `for`, and `from` recorded 0% recall due to extreme dataset imbalance in Visual Genome.
2. **Ontology View Projection Drop**: When projected to `strict` (6 predicates), `extended` (8 predicates), or `tri_dataset` (20 predicates), Recall drops sharply (R@100: 3.58% on strict, 0.94% on extended). This occurs because VG predicates such as `on`, `has`, and `wearing` are either mapped to a narrow subset or constrained by strict mutual exclusivity in the normalized target ontologies.

---

## 6. Data Integrity Verification

Post-execution cryptographic SHA-256 verification was conducted across all 851 files in `evaluation_sets/` against the baseline established in Prompt 1 (`results/REACT/eval_sets_sha256_post.json`):
- **Files Verified**: 851 / 851
- **Integrity Violations**: 0
- **Modified / Deleted / Added Files**: 0
- **Status**: **100% BIT-FOR-BIT IDENTICAL** to baseline.

The inventory snapshot is preserved in `results/FastContextualSGG/eval_sets_sha256_post.json`.

---

## 7. Deliverables & Artifact Locations

All deliverables are saved outside `evaluation_sets/`:
- **Compatibility Report**: [`results/FastContextualSGG/compatibility_report.md`](file:///d:/Minor/results/FastContextualSGG/compatibility_report.md)
- **Machine-Readable Metrics (JSON)**: [`results/FastContextualSGG/metrics.json`](file:///d:/Minor/results/FastContextualSGG/metrics.json)
- **Machine-Readable Metrics (CSV)**: [`results/FastContextualSGG/metrics.csv`](file:///d:/Minor/results/FastContextualSGG/metrics.csv)
- **Run Configuration**: [`results/FastContextualSGG/run_config.json`](file:///d:/Minor/results/FastContextualSGG/run_config.json)
- **Execution Log**: [`results/FastContextualSGG/run.log`](file:///d:/Minor/results/FastContextualSGG/run.log)
- **Cryptographic SHA-256 Inventory**: [`results/FastContextualSGG/eval_sets_sha256_post.json`](file:///d:/Minor/results/FastContextualSGG/eval_sets_sha256_post.json)
- **Scene Graph Visualizations**:
  - [`results/FastContextualSGG/visualizations/vg_2315564_c-sgg_pred.jpg`](file:///d:/Minor/results/FastContextualSGG/visualizations/vg_2315564_c-sgg_pred.jpg)
  - [`results/FastContextualSGG/visualizations/vg_2315563_c-sgg_pred.jpg`](file:///d:/Minor/results/FastContextualSGG/visualizations/vg_2315563_c-sgg_pred.jpg)
  - [`results/FastContextualSGG/visualizations/vg_2315564_cv-sgg_pred.jpg`](file:///d:/Minor/results/FastContextualSGG/visualizations/vg_2315564_cv-sgg_pred.jpg)
  - [`results/FastContextualSGG/visualizations/vg_2315563_cv-sgg_pred.jpg`](file:///d:/Minor/results/FastContextualSGG/visualizations/vg_2315563_cv-sgg_pred.jpg)
- **Adapter & Tests**:
  - Adapter: [`experiments/FastContextualSGG/fast_contextual_adapter.py`](file:///d:/Minor/experiments/FastContextualSGG/fast_contextual_adapter.py)
  - Unit Tests: [`experiments/FastContextualSGG/test_fast_contextual_evaluator.py`](file:///d:/Minor/experiments/FastContextualSGG/test_fast_contextual_evaluator.py)
  - Benchmark Runner: [`experiments/FastContextualSGG/run_benchmark_fast_contextual.py`](file:///d:/Minor/experiments/FastContextualSGG/run_benchmark_fast_contextual.py)
