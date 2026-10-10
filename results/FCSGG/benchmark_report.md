# Comprehensive Benchmark Report: Pretrained FCSGG Evaluation

**Research Project**: *Object Retrieval in Dynamic Workspace Environments*  
**Model Under Evaluation**: FCSGG (Fully Convolutional Scene Graph Generation, CVPR 2021)  
**Authors**: Hengyue Liu, Ning Yan, Masood Mortazavi, Bir Bhanu (Futurewei Technologies Inc. / UC Riverside)  
**Execution Environment**: Local CPU (`Intel(R) Core(TM) Ultra 5 125H`, 18 threads, PyTorch 2.8.0+cpu)  
**Evaluation Scope**: 200-image frozen benchmark subset of Visual Genome (VG150)  
**Evaluated Ontology Views**: `strict` (6 predicates), `extended` (8 predicates), `tri_dataset` (20 predicates), `native` (50 predicates)  
**Task Mode**: Scene Graph Detection (`SGDet`)  

---

## 1. Executive Summary & Evaluation Integrity

This report documents the rigorous evaluation of the pretrained **FCSGG** model on the standardized benchmarking framework of the *Object Retrieval in Dynamic Workspace Environments* research project. This benchmark follows Prompt 1 (REACT on PSG and VG150) and Prompt 2 (REACT++ on PSG).

### Core Protocol Adherence & Safety
1. **Strict Read-Only Isolation**: The `evaluation_sets/` directory remained 100% read-only throughout all test design, adapter validation, smoke tests, and full evaluations. A post-run SHA-256 cryptographic audit over all **851 files** confirmed zero bit-level modifications (`eval_sets_sha256_post.json` $\equiv$ `eval_sets_sha256_pre.json`).
2. **Defensible Dataset Matrix**:
   - **VG150 Evaluated**: Pretrained FCSGG is natively configured on Visual Genome (150 object classes, 50 predicate classes). Evaluated across all 4 standardized ontology views (`strict`, `extended`, `tri_dataset`, `native`).
   - **PSG Blocked**: Disjoint object vocabulary (133 COCO thing/stuff classes vs. 150 VG classes). Cross-dataset SGDet is scientifically invalid without joint object-head alignment.
   - **IndoorVG Blocked**: 100% training-split contamination (`LEAK_INDOOR_VGTRAIN`) with the VG150 training set.
   - **Open Images V6 Blocked**: Disjoint object vocabulary (601 classes).
3. **Preservation of Native FCSGG Formulation**:
   - Rather than forcing FCSGG into a two-stage RoI pooling or Transformer architecture, the evaluation preserves its native **fully convolutional CenterNet + Relation Affinity Field (RAF)** design.
   - Bounding boxes are decoded from predicted center keypoint heatmaps and scale regressions.
   - Directed relationships are scored via line-integral path sampling along the predicted 2D vector field connecting subject and object center locations.
4. **Standardized Evaluator Validation**:
   - Four dedicated unit tests ([test_fcsgg_evaluator.py](file:///d:/Minor/experiments/FCSGG/test_fcsgg_evaluator.py)) verified coordinate preservation, 1-indexed class decoding, directed pair ranking, and ontology projection before launching inference.
   - A 3-image pre-flight smoke test confirmed end-to-end execution sanity.

---

## 2. Quantitative Benchmark Results

All evaluations were conducted in **SGDet** mode (detecting bounding boxes, classifying objects, and predicting directed relationship triplets simultaneously). Metrics are reported under the project's dual empty-ground-truth policy:
1. **Eligible Image Recall ($R@K$)**: Computed over images containing $\ge 1$ ground truth relation in that view.
2. **All-200 Image Sensitivity Recall ($R@K_{\text{all200}}$)**: Evaluated across all 200 images, penalizing empty ground-truth images with $0.0$.
3. **Mean Recall ($mR@K$)**: Unweighted macro-average across all active predicate classes present in ground truth.

### Quantitative Summary Table

| Model | Checkpoint | Dataset | Ontology View | Task | Eligible Imgs | Det mAP@50 | R@20 (Elig.) | R@50 (Elig.) | R@100 (Elig.) | R@50 (All-200) | R@100 (All-200) | mR@20 | mR@50 | mR@100 | E2E Latency (ms) | FPS | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FCSGG** | `HRNetW32-1S.pth` | **VG150** | `strict` (6) | SGDet | 141 / 200 | **0.2253** | **0.0203** | **0.0238** | **0.0262** | 0.0168 | 0.0185 | 0.0049 | **0.0082** | **0.0142** | **1277.1** | **0.78** | **3160.6** |
| **FCSGG** | `HRNetW32-1S.pth` | **VG150** | `extended` (8) | SGDet | 142 / 200 | **0.2253** | **0.0192** | **0.0227** | **0.0227** | 0.0161 | 0.0161 | 0.0040 | **0.0069** | **0.0069** | **1277.1** | **0.78** | **3160.6** |
| **FCSGG** | `HRNetW32-1S.pth` | **VG150** | `tri_dataset` (20)| SGDet | 160 / 200 | **0.2253** | **0.0108** | **0.0128** | **0.0146** | 0.0103 | 0.0117 | 0.0011 | **0.0022** | **0.0051** | **1277.1** | **0.78** | **3160.6** |
| **FCSGG** | `HRNetW32-1S.pth` | **VG150** | `native` (50) | SGDet | 174 / 200 | **0.2253** | **0.0699** | **0.0968** | **0.1223** | 0.0842 | 0.1064 | 0.0222 | **0.0268** | **0.0326** | **1277.1** | **0.78** | **3160.6** |

---

## 3. Comparative Architecture & Performance Analysis: REACT vs. FCSGG on VG150

Both models were evaluated on the exact same frozen 200-image Visual Genome subset under identical SGDet evaluation metrics:

| Metric / Dimension | REACT (YOLOv8m) | FCSGG (CenterNet / HRNetW32) | Comparative Analysis |
| :--- | :--- | :--- | :--- |
| **Architectural Paradigm** | Two-stage candidate pairing + Context Transformer | Fully convolutional one-stage CenterNet + 2D RAF vector field | Bottom-up point regression vs. Region-based proposal heads |
| **Detector mAP@50 (VG150)** | **0.3758** (37.58%) | **0.2253** (22.53%) | YOLOv8m achieves higher detection precision on dense VG boxes |
| **Native VG150 R@50 (Elig.)** | **0.2455** (24.55%) | **0.0968** (9.68%) | Compounding detection misses impact bottom-up triplet pairing |
| **Native VG150 R@100 (Elig.)** | **0.2777** (27.77%) | **0.1223** (12.23%) | FCSGG captures prominent structural relations (`has` 33.3%, `wearing` 21.7%) |
| **Strict View R@100 (Elig.)** | 0.0096 (0.96%) | **0.0262** (2.62%) | FCSGG shows higher sensitivity on strict contact relations (`on`, `wearing`) |
| **Mean End-to-End Latency** | **701.8 ms** | **1277.1 ms** | YOLOv8m forward pass is 1.8x faster on CPU |
| **Throughput (FPS)** | **1.42 FPS** | **0.78 FPS** | REACT achieves higher CPU throughput |
| **Peak RAM (MB)** | 3506.2 MB | **3160.6 MB** | FCSGG has 345 MB lower resident memory footprint |

### In-Depth Findings

#### Finding 1: CenterNet Detection Sensitivity on Dense VG Annotations
In Visual Genome, annotations feature extreme bounding box density, occlusions, and severe aspect ratio variations. CenterNet models objects via a single center keypoint per instance. When multiple objects share nearby centers or overlapping regions (e.g., person and shirt), center heatmap collapse and coarse $(w, h)$ regression lead to lower detector precision (mAP@50 = 22.53% vs. 37.58% for YOLOv8m). Because SGDet strictly requires both subject and object bounding boxes to achieve $\text{IoU} \ge 0.5$ simultaneously, lower detection precision creates an immediate ceiling on true-positive triplet matching.

#### Finding 2: Relation Affinity Field Mechanics & Path Integral Discretization
FCSGG eliminates $O(N^2)$ candidate pair feature pooling by computing relation scores via line-integral path integration along the predicted vector field connecting center points.
- On large, well-separated objects, RAF integration reliably predicts dominant relations (`has`: **33.35%**, `wearing`: **21.71%**, `riding`: **20.00%**).
- For spatially distant or heavily occluded pairs, path integration across downsampled feature grids ($4\times$ stride) accumulates discretization error, causing lower recall on spatial tail predicates (`under`: 0.0%, `behind`: 0.0%).

#### Finding 3: Behavior Under Compact Ontologies
Visual Genome ground truth is predominantly structural (`has`, `of`, `part of`, `sitting on`). None of these appear in the 6-predicate `strict` ontology.
- While REACT suffered heavy candidate filtering when dropping from 50 native predicates to 6 strict predicates ($R@100 = 0.96\%$), FCSGG retained a higher proportion of its valid detections on `on` (2.9%), `wearing` (2.5%), and `in` (3.1%), achieving **$R@100 = 2.62\%$** on the strict view.

---

## 4. Per-Predicate Recall Analysis (K=100)

### A. Strict and Extended Views on VG150

| Predicate | Ground Truth Count | Strict Recall@100 | Extended Recall@100 | Semantic Category |
| :--- | :--- | :--- | :--- | :--- |
| **in** | 76 | **0.0312** | 0.0000 | Spatial / Enclosure |
| **on** | 364 | **0.0292** | **0.0280** | Spatial / Support (Head) |
| **wearing** | 121 | **0.0250** | **0.0200** | Interactive / Attire |
| **eating** | 2 | 0.0000 | 0.0000 | Interactive / Action |
| **hanging from** | 5 | 0.0000 | 0.0000 | Spatial / Suspension |
| **holding** | 36 | 0.0000 | 0.0000 | Interactive / Grasping |
| **looking at** | 1 | — | 0.0000 | Attention / Gaze |
| **playing** | 0 | — | 0.0000 | Leisure / Activity |

### B. Top Performing Native VG150 Predicates (K=100)

| Predicate | Ground Truth Count | Native Recall@100 | Semantic Category |
| :--- | :--- | :--- | :--- |
| **has** | 149 | **0.3335** | Compositional / Structural |
| **wearing** | 113 | **0.2171** | Attire / Interactive |
| **riding** | 8 | **0.2000** | Interactive / Locomotion |
| **on** | 364 | **0.1348** | Spatial / Contact (Head) |
| **in** | 76 | **0.0977** | Spatial / Enclosure |
| **of** | 77 | **0.0940** | Attributive / Part-Whole |
| **holding** | 36 | **0.0588** | Grasping / Manipulation |
| **with** | 46 | **0.0370** | Associative / Proximity |
| **near** | 49 | **0.0345** | Spatial Proximity |

---

## 5. System Profiling & Latency Breakdown

Profiling conducted during live 200-image evaluation on the host CPU:

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| **Host Processor** | Intel(R) Core(TM) Ultra 5 125H (14 cores / 18 threads) | Hardware |
| **Execution Framework** | PyTorch 2.8.0+cpu | Software |
| **Input Image Resolution** | $512 \times 512$ (shortest-edge resized) | Resolution |
| **Batch Size** | 1 | Samples |
| **Forward Pass Latency** | **$1271.7 \pm 112.4$** | ms |
| **Total End-to-End Latency** | **$1277.1 \pm 114.8$** | ms |
| **Throughput** | **$0.78$** | FPS |
| **Peak Resident RAM (RSS)** | **$3160.6$** | MB |

---

## 6. Cryptographic Data Integrity Audit

In compliance with the workspace protection mandate, SHA-256 digests across all 851 files in `evaluation_sets/` were verified post-execution against the baseline established during Prompt 1:

- **Baseline Digest File**: `file:///d:/Minor/results/REACT/eval_sets_sha256_post.json` (851 files)
- **Post-Run Digest File**: `file:///d:/Minor/results/FCSGG/eval_sets_sha256_post.json` (851 files)
- **Files Checked**: **851**
- **Files Modified**: **0**
- **Files Added**: **0**
- **Files Deleted**: **0**
- **Status**: **100% BIT-FOR-BIT IDENTICAL & UNMODIFIED**

---

## 7. Deliverables & Artifact Index

All generated code, logs, metrics, and visual artifacts reside outside `evaluation_sets/`:

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **Benchmark Report** | [benchmark_report.md](file:///d:/Minor/results/FCSGG/benchmark_report.md) | Full evaluation report with extended analysis |
| **Compatibility Report** | [compatibility_report.md](file:///d:/Minor/results/FCSGG/compatibility_report.md) | Architectural analysis, RAF mechanics, and dataset compatibility matrix |
| **Machine-Readable Metrics (JSON)** | [metrics.json](file:///d:/Minor/results/FCSGG/metrics.json) | Complete metric hierarchy including per-predicate recall and distributions |
| **Machine-Readable Metrics (CSV)** | [metrics.csv](file:///d:/Minor/results/FCSGG/metrics.csv) | Standardized CSV table matching the Prompt 1 & 2 schema |
| **Run Configuration** | [run_config.json](file:///d:/Minor/results/FCSGG/run_config.json) | Hardware profile, PyTorch runtime settings, and reproducibility metadata |
| **Execution Trace** | [run.log](file:///d:/Minor/results/FCSGG/run.log) | Complete timestamped log of smoke test and full 200-image evaluation |
| **Data Integrity Inventory** | [eval_sets_sha256_post.json](file:///d:/Minor/results/FCSGG/eval_sets_sha256_post.json) | Post-execution SHA-256 cryptographic audit of all 851 evaluation files |
| **Visualizations** | [VG150_2315564_pred.jpg](file:///d:/Minor/results/FCSGG/visualizations/VG150_2315564_pred.jpg) | Qualitative prediction visualization for sample 2315564 |
| **Visualizations** | [VG150_2315563_pred.jpg](file:///d:/Minor/results/FCSGG/visualizations/VG150_2315563_pred.jpg) | Qualitative prediction visualization for sample 2315563 |
| **Inference Adapter** | [fcsgg_adapter.py](file:///d:/Minor/experiments/FCSGG/fcsgg_adapter.py) | Standardized FCSGG adapter preserving CenterNet coordinates and RAFs |
| **Evaluator Unit Tests** | [test_fcsgg_evaluator.py](file:///d:/Minor/experiments/FCSGG/test_fcsgg_evaluator.py) | Unit tests verifying coordinate preservation, indexing, directionality, and projection |
| **Benchmark Runner** | [run_benchmark_fcsgg.py](file:///d:/Minor/experiments/FCSGG/run_benchmark_fcsgg.py) | End-to-end benchmark execution pipeline |

### Exact Reproduction Command
```powershell
python d:/Minor/experiments/FCSGG/run_benchmark_fcsgg.py
```
