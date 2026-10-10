# Comprehensive Benchmark Report: Pretrained REACT++ Evaluation

**Research Project**: *Object Retrieval in Dynamic Workspace Environments*  
**Model**: REACT++ (YOLO12m ONNX, SGG-Benchmark / Maelic)  
**Execution Environment**: Local CPU (`Intel(R) Core(TM) Ultra 5 125H`, 18 threads, ONNX Runtime 1.27.0 `CPUExecutionProvider`)  
**Evaluation Scope**: 200-image frozen benchmark subset of Panoptic Scene Graph (PSG)  
**Evaluated Ontology Views**: `strict` (6 predicates), `extended` (8 predicates), `tri_dataset` (20 predicates), `native` (56 predicates)  
**Task Mode**: Scene Graph Detection (`SGDet`)  

---

## 1. Executive Summary & Evaluation Integrity

This report documents the rigorous benchmarking of the **REACT++** (YOLO12m ONNX) model under the standardized benchmarking protocol of the *Object Retrieval in Dynamic Workspace Environments* project. It directly follows Prompt 1 (which benchmarked PyTorch REACT on PSG and VG150).

### Core Protocol Adherence & Safety
1. **Strict Read-Only Isolation**: The `evaluation_sets/` directory remained 100% read-only throughout all test design, smoke tests, and full evaluations. Cryptographic SHA-256 auditing across all **851 files** confirmed zero bit-level modifications (`eval_sets_sha256_post.json`).
2. **Defensible Dataset Matrix**:
   - **PSG Evaluated**: The available official checkpoint (`checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx`) is natively trained on the 133-object-category and 56-predicate Panoptic Scene Graph (PSG) dataset. It was evaluated across all four standardized ontology views (`strict`, `extended`, `tri_dataset`, `native`).
   - **Cross-Dataset SGDet Blocked**: Evaluating a 133-class detector against VG150 (150 disjoint classes) or Open Images V6 (601 disjoint classes) in SGDet mode without an aligned multi-dataset object head is scientifically invalid, as triplets require both subject and object detection at IoU $\ge 0.5$.
   - **IndoorVG Blocked**: Excluded due to 100% training-split contamination (`LEAK_INDOOR_VGTRAIN`) with VG150 and non-overlapping object semantics with PSG.
3. **Native Predicate Vocabulary Decoding**: Raw ONNX outputs contain 57 relation output dimensions (`__background__` + 56 native predicates). In accordance with the prompt's explicit guidance, predictions were decoded using their native vocabulary before ontology projection, preventing artificial truncation to 6 classes at model output time.
4. **Evaluator Unit Test Suite**: Four dedicated regression tests (`experiments/REACTPP/test_reactpp_evaluator.py`) verified native decoding, multi-view projection, subject $\to$ object directionality, and invalid dataset rejection before any evaluation began.
5. **Pre-Flight Smoke Test**: A 3-image dry run confirmed bounding box scaling, unletterbox transformation, and metric computation before running the full 200-image benchmark.

---

## 2. Quantitative Benchmark Results

All evaluations were conducted in **SGDet** mode (detecting bounding boxes, classifying objects, and predicting directed relationship triplets simultaneously). Metrics are reported under the project's dual empty-ground-truth policy:
1. **Eligible Image Recall ($R@K$)**: Computed over images containing $\ge 1$ ground truth relation in that view.
2. **All-200 Image Sensitivity Recall ($R@K_{\text{all200}}$)**: Evaluated across all 200 images, penalizing empty ground-truth images with $0.0$.
3. **Mean Recall ($mR@K$)**: Unweighted macro-average across all active predicate classes present in ground truth.

### Quantitative Summary Table

| Model | Checkpoint | Dataset | Ontology View | Task | Eligible Imgs | Det mAP@50 | R@20 (Elig.) | R@50 (Elig.) | R@100 (Elig.) | R@50 (All-200) | R@100 (All-200) | mR@20 | mR@50 | mR@100 | E2E Latency (ms) | FPS | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REACT++** | `yolo12m.onnx` | **PSG** | `strict` (6) | SGDet | 173 / 200 | **0.5951** | **0.3222** | **0.3351** | **0.3351** | 0.2899 | 0.2899 | 0.4271 | **0.4790** | **0.4790** | **548.0** | **1.82** | **789.3** |
| **REACT++** | `yolo12m.onnx` | **PSG** | `extended` (8) | SGDet | 176 / 200 | **0.5951** | **0.3003** | **0.3140** | **0.3140** | 0.2763 | 0.2763 | 0.3614 | **0.4033** | **0.4033** | **548.0** | **1.82** | **789.3** |
| **REACT++** | `yolo12m.onnx` | **PSG** | `tri_dataset` (20)| SGDet | 198 / 200 | **0.5951** | **0.2561** | **0.3053** | **0.3130** | 0.3022 | 0.3099 | 0.2663 | **0.3076** | **0.3137** | **548.0** | **1.82** | **789.3** |
| **REACT++** | `yolo12m.onnx` | **PSG** | `native` (56) | SGDet | 200 / 200 | **0.5951** | **0.3135** | **0.3598** | **0.3779** | 0.3598 | 0.3779 | 0.2088 | **0.2320** | **0.2376** | **548.0** | **1.82** | **789.3** |

---

## 3. Comparative Architecture & Performance Analysis: REACT vs. REACT++

| Metric / Characteristic | REACT (Prompt 1) | REACT++ (Prompt 2) | Relative Change / Impact |
| :--- | :--- | :--- | :--- |
| **Backbone & Architecture** | YOLOv8m + PyTorch SGG Head | YOLO12m + ONNX End-to-End Graph | Modernized YOLO12 architecture |
| **Runtime & Execution Provider** | PyTorch 2.8.0 CPU | ONNX Runtime 1.27.0 `CPUExecutionProvider` | Fused graph, constant folding |
| **Mean Forward Latency** | $1215.7$ ms | **$527.7$ ms** | **2.30x speedup** |
| **Mean End-to-End Latency** | $1223.9$ ms | **$548.0$ ms** | **2.23x speedup** |
| **Throughput (FPS)** | $0.82$ FPS | **$1.82$ FPS** | **+122% throughput** |
| **Peak Resident RAM** | $3798.7$ MB | **$789.3$ MB** | **4.81x lower memory footprint** |
| **Object Detection mAP@50** | $0.5673$ (56.73%) | **$0.5951$ (59.51%)** | **+2.78% absolute gain** |
| **Native PSG R@50 (Eligible)** | $0.3047$ (30.47%) | **$0.3598$ (35.98%)** | **+5.51% absolute gain** |
| **Native PSG R@100 (Eligible)** | $0.3394$ (33.94%) | **$0.3779$ (37.79%)** | **+3.85% absolute gain** |
| **Strict View R@50 (Eligible)** | $0.5162$ (51.62%) | $0.3351$ (33.51%) | Triplet candidate saturation effect |
| **Strict View mR@50** | $0.5214$ (52.14%) | $0.4790$ (47.90%) | Balanced predicate representation |

### In-Depth Findings

#### Finding 1: Superior Detector Accuracy & Native SGG Recall
REACT++ benefits from YOLO12m's improved feature pyramid and receptive field scaling. Detection mAP@50 increases from **56.73% to 59.51%**, which directly translates into higher native scene graph recall on PSG:
- **Native R@50**: increases from **30.47% to 35.98%** (+18.1% relative improvement).
- **Native R@100**: increases from **33.94% to 37.79%** (+11.3% relative improvement).

#### Finding 2: The Top-100 Triplet Restriction & Ontology Projection Saturation
In the exported ONNX model (`react_pp_yolo12m.onnx`), the relation prediction head outputs top triplet candidates ranked by combined object-relation score up to a maximum cap ($K=100$ per image).
- When evaluating on **Native PSG (56 predicates)**, recall scales normally from $R@20 = 31.35\%$ to $R@50 = 35.98\%$ and $R@100 = 37.79\%$.
- However, when projected down onto the **Strict View (6 predicates)**, only the subset of the top-100 relations that match the 6 strict predicates survive projection. On average, fewer than 50 relations per image survive projection, causing $R@50$ and $R@100$ to saturate at identical values (**$33.51\%$**).
- Despite this candidate ceiling, REACT++ maintains strong tail predicate recall, achieving **$80.0\%$ on `eating`** and **$54.0\%$ on `hanging from`**.

#### Finding 3: 2.2x Faster Latency and 4.8x Reduced Memory on CPU
The ONNX Runtime execution graph drastically reduces overhead compared to the PyTorch implementation:
- End-to-end CPU latency drops from $1224$ ms to **$548$ ms per image**.
- Peak memory usage drops from $3.8$ GB to **$789$ MB**, enabling deployment on resource-constrained robotics and dynamic workspace edge devices.

---

## 4. Detailed Per-Predicate Recall Analysis (K=100)

### A. Strict and Extended Views

| Predicate | Ground Truth Count | Strict Recall@100 | Extended Recall@100 | Tri-Dataset Recall@100 | Category Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **eating** | 10 | **0.8000** | **0.8000** | — | Interactive / Tail |
| **hanging from** | 33 | **0.5400** | **0.5400** | **0.5400** | Spatial / Support |
| **holding** | 72 | **0.4927** | **0.4927** | **0.4927** | Interactive / Body |
| **in** | 43 | **0.2862** | **0.2862** | **0.2862** | Spatial / Enclosure |
| **on** | 343 | **0.2810** | **0.2810** | **0.2810** | Spatial / Support (Head) |
| **wearing** | 19 | **0.4744** | **0.4744** | **0.4744** | Interactive / Attire |
| **looking at** | 51 | — | **0.1022** | **0.1022** | Attentive / Gaze |
| **playing** | 4 | — | **0.2500** | **0.2500** | Interactive / Tail |

### B. Top Performing Native PSG Predicates (K=100)

| Predicate | Ground Truth Count | Native Recall@100 | Semantic Category |
| :--- | :--- | :--- | :--- |
| **about to hit** | 1 | **1.0000** | Dynamic Action |
| **driving on** | 53 | **0.8088** | Motion / Vehicle |
| **over** | 109 | **0.7133** | Spatial / Elevation |
| **riding** | 12 | **0.6875** | Transportation / Interaction |
| **driving** | 7 | **0.5714** | Transportation / Control |
| **hanging from** | 33 | **0.5400** | Spatial / Suspension |
| **parked on** | 40 | **0.5370** | Static / Placement |
| **walking on** | 29 | **0.5217** | Motion / Locomotion |
| **sitting on** | 29 | **0.5000** | Body Posture |
| **slicing** | 3 | **0.5000** | Tool Interaction |
| **playing** | 2 | **0.5000** | Leisure / Activity |
| **holding** | 72 | **0.4927** | Grasping / Manipulation |
| **wearing** | 19 | **0.4744** | Body / Attire |
| **crossing** | 14 | **0.4444** | Locomotion / Transiting |
| **standing on** | 53 | **0.3943** | Body Posture / Contact |
| **beside** | 171 | **0.3814** | Spatial Proximity |
| **feeding** | 3 | **0.3333** | Care / Interaction |
| **carrying** | 23 | **0.3158** | Transporting |
| **in front of** | 104 | **0.3086** | Spatial Occlusion |

---

## 5. System Profiling & Latency Breakdown

Profiling conducted during live 200-image evaluation on the host CPU:

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| **Host Processor** | Intel(R) Core(TM) Ultra 5 125H (14 cores / 18 threads) | Hardware |
| **Execution Provider** | ONNX Runtime `CPUExecutionProvider` | Software |
| **Input Resolution** | $640 \times 640$ (letterboxed with 114 padding) | Image Dimension |
| **Batch Size** | 1 | Samples |
| **Pre-processing Time** | $12.1 \pm 3.4$ | ms |
| **Model Forward Pass** | **$527.7 \pm 42.1$** | ms |
| **Post-processing & Unletterbox** | $8.2 \pm 2.1$ | ms |
| **Total End-to-End Latency** | **$548.0 \pm 44.8$** | ms |
| **Inference Throughput** | **$1.82$** | FPS |
| **Peak Resident Set Size (RSS)** | **$789.3$** | MB |

---

## 6. Cryptographic Data Integrity Audit

In compliance with the workspace protection mandate, SHA-256 digests across all 851 files in `evaluation_sets/` were verified post-execution against the baseline established during Prompt 1:

- **Baseline Digest File**: `file:///d:/Minor/results/REACT/eval_sets_sha256_post.json` (851 files)
- **Post-Run Digest File**: `file:///d:/Minor/results/REACTPP/eval_sets_sha256_post.json` (851 files)
- **Files Checked**: **851**
- **Files Modified**: **0**
- **Files Added**: **0**
- **Files Deleted**: **0**
- **Status**: **100% BIT-FOR-BIT IDENTICAL & UNMODIFIED**

---

## 7. Deliverables & Artifact Index

All generated code, logs, metrics, and visual artifacts reside outside `evaluation_sets/`:

1. [compatibility_report.md](file:///d:/Minor/results/REACTPP/compatibility_report.md): Inspection report covering ONNX signatures, task compatibility, and dataset constraints.
2. [benchmark_report.md](file:///d:/Minor/results/REACTPP/benchmark_report.md): This comprehensive quantitative and qualitative benchmark report.
3. [metrics.csv](file:///d:/Minor/results/REACTPP/metrics.csv): Standardized machine-readable CSV table of benchmark results.
4. [metrics.json](file:///d:/Minor/results/REACTPP/metrics.json): Complete metric hierarchy including per-predicate recall and distributions.
5. [run_config.json](file:///d:/Minor/results/REACTPP/run_config.json): Reproducibility metadata, hardware profile, and runtime configuration.
6. [run.log](file:///d:/Minor/results/REACTPP/run.log): Complete timestamped execution log.
7. [visualizations/](file:///d:/Minor/results/REACTPP/visualizations): Sample detected scene graph prediction visualizations:
   - [PSG_107907_pred.jpg](file:///d:/Minor/results/REACTPP/visualizations/PSG_107907_pred.jpg)
   - [PSG_107910_pred.jpg](file:///d:/Minor/results/REACTPP/visualizations/PSG_107910_pred.jpg)
8. [test_reactpp_evaluator.py](file:///d:/Minor/experiments/REACTPP/test_reactpp_evaluator.py): Unit test suite for native decoding, projection, and directional integrity.
9. [run_benchmark_reactpp.py](file:///d:/Minor/experiments/REACTPP/run_benchmark_reactpp.py): Benchmark execution pipeline.
