# Object Retrieval in Dynamic Workspace Environments

[![Python 3.13](https://img.shields.io/badge/Python-3.13.7-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8.0%2Bcpu-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.27.0-005ced.svg)](https://onnxruntime.ai/)
[![Dataset Integrity](https://img.shields.io/badge/Data%20Audit-851%2F851%20SHA--256%20Verified-brightgreen.svg)]()
[![Benchmark Task](https://img.shields.io/badge/Task-SGDet%20(IoU%20%E2%89%A5%200.5)-orange.svg)]()

> **Research Project**: *Object Retrieval in Dynamic Workspace Environments*  
> **Host Testbed**: Intel(R) Core(TM) Ultra 5 125H (14 Cores / 18 Threads, 16 GB RAM [15.61 GB visible], Windows 11)  
> **Master Reports**: [`final_report.md`](final_report.md) | [`METHODOLOGY_CROSS_DATASET_BENCHMARKING.md`](METHODOLOGY_CROSS_DATASET_BENCHMARKING.md)  
> **Audit & Matrix Records**: [`results/audit_discrepancy_log.md`](results/audit_discrepancy_log.md) | [`results/single_pair_benchmark_matrix.csv`](results/single_pair_benchmark_matrix.csv) | [`results/reproducibility_audit_records.json`](results/reproducibility_audit_records.json)  

---
## 📌 Executive Overview

In dynamic workspace environments (collaborative robotic workspaces, tabletop object manipulation, and indoor retrieval), agents must perceive not merely isolated object bounding boxes, but also directed structural, physical, and functional relationships (e.g., `drill on table`, `hand holding screwdriver`, `book in drawer`).

This repository provides an empirical, standardized benchmark suite evaluating **four state-of-the-art scene graph generation (SGG) model architectures** under identical hardware conditions, evaluating on frozen 200-image evaluation subsets across four benchmark datasets:
1. **Panoptic Scene Graph (PSG)**: 200 images, 3,019 objects, 1,385 relations, 133 COCO classes, 56 OpenPSG predicates.
2. **Visual Genome (VG150)**: 200 images, 2,320 objects, 1,083 relations, 150 object classes, 50 predicate classes.
3. **IndoorVG**: 200 images, 3,118 objects, 1,672 relations, 84 indoor object categories, 31 indoor predicates.
4. **OpenImages V6**: 200 images, 1,486 objects, 866 relations (146 physical relations, 720 visual attributes), 75 object categories, 31 relationship predicates.

### Evaluated Model Architectures (16 Core Combinations)
1. **REACT** (*Relation-Enhanced Architecture with Context-guided Transformers*, BMVC 2025): PyTorch-based YOLOv8m detector coupled with context-guided Transformer relation head.
2. **REACT++** (*Fast SGG with Attention-driven Feature Interaction*, SGG-Benchmark / Maelic): Modernized YOLO12m end-to-end graph compiled to ONNX Runtime.
3. **FCSGG** (*Fully Convolutional Scene Graph Generation*, CVPR 2021): Single-stage bottom-up architecture combining CenterNet object keypoint regression with 2D **Relation Affinity Fields (RAF)**.
4. **Fast Contextual SGG (C-SGG)** (*Fast Contextual SGG with Unbiased Context Augmentation*, Jin et al.): Context-first architecture utilizing a lightweight Context Knowledge Network (CKN, 3-layer MLP) with GloVe 50d semantics and 2D spatial geometries.

*(Note: CV-SGG is evaluated as an additional two-stage visual-verification variant on VG150 and is presented separately from the 16 core combinations below).*

---

## 📊 Complete 16-Combination Benchmark Matrix (4 Models × 4 Datasets)

All models were evaluated under standardized **Scene Graph Detection (`SGDet`)** criteria (bounding box IoU ≥ 0.5 on both subject and object, with correct category and directed predicate predictions):

| Run ID | Model Architecture | Evaluated Variant | Exact Detector / Runtime | Target Dataset | Regime | Image-Avg R@50 | Image-Avg R@100 | Pooled R@50 | Pooled R@100 | Image-Avg mR@50 | Image-Avg mR@100 | Det. mAP@50 | Det. H(AP,AR) [Supp] | Supp. H(R,mR)@50 | Latency (ms) | Speed (FPS) | Cache Status |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RUN-01** | **REACT** | PyTorch YOLOv8m | YOLOv8m (PyTorch CPU) | **PSG** | In-Domain | 30.47% | 33.94% | 30.11% | 33.86% | 23.98% | 26.03% | 56.73% | 62.14% | 26.84% | 1,223.9 ms | 0.82 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-02** | **REACT** | PyTorch YOLOv8m | YOLOv8m (PyTorch CPU) | **VG150** | In-Domain | 24.55% | 27.77% | 26.04% | 28.53% | 16.61% | 18.28% | 37.58% | 45.46% | 19.81% | 701.8 ms | 1.42 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-03** | **REACT** | ONNX YOLOv8m (Native) | YOLOv8m ONNX Runtime | **IndoorVG** | In-Domain | 31.08% | 36.58% | 29.96% | 35.89% | 26.48% | 29.65% | 49.31% | 57.98% | 28.60% | 724.8 ms | 1.38 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-04** | **REACT** | PyTorch YOLOv8m | YOLOv8m (PyTorch CPU) | **OpenImages V6** | Zero-Shot Transfer | 1.78% | 1.90% | 2.08% | 2.31% | 10.76% | 12.05% | 17.51% | 21.53% | 3.05% | 1,184.2 ms | 0.84 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-05** | **REACT++** | ONNX YOLO12m | YOLO12m ONNX Runtime | **PSG** | In-Domain | **35.98%** | **37.79%** | **35.60%** | **37.40%** | **23.20%** | **23.76%** | **59.51%** | **62.84%** | **28.21%** | **548.0 ms** | **1.82 FPS** | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-06** | **REACT++** | ONNX YOLO12m | YOLO12m ONNX Runtime | **VG150** | In-Domain | 23.54% | 25.56% | 26.59% | 28.62% | 16.57% | 18.48% | 37.43% | 46.68% | 19.45% | 744.0 ms | 1.34 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-07** | **REACT++** | ONNX YOLO12m | YOLO12m ONNX Runtime | **IndoorVG** | Zero-Shot Transfer | 13.64% | 16.02% | 15.19% | 17.46% | 12.69% | 14.07% | 21.50% | 26.88% | 13.15% | 2,450.7 ms | 0.41 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-08** | **REACT++** | ONNX YOLO12m | YOLO12m ONNX Runtime | **OpenImages V6** | Zero-Shot Transfer | 2.12% | 2.24% | 2.19% | 2.31% | 13.22% | 14.11% | 17.91% | 22.77% | 3.66% | 599.3 ms | 1.67 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-09** | **FCSGG** | CenterNet / RAF | HRNetW32 CenterNet (PyTorch) | **PSG** | Zero-Shot Transfer | 4.39% | 5.10% | 4.33% | 5.05% | 1.10% | 1.40% | 17.45% | 21.23% | 1.76% | 1,722.3 ms | 0.58 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-10** | **FCSGG** | CenterNet / RAF | HRNetW32 CenterNet (PyTorch) | **VG150** | In-Domain | 9.08% | 12.23% | 9.14% | 12.28% | 2.62% | 3.26% | 22.53% | 31.62% | 4.06% | 1,277.1 ms | 0.78 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-11** | **FCSGG** | CenterNet / RAF | HRNetW32 CenterNet (PyTorch) | **IndoorVG** | Zero-Shot Transfer | 7.97% | 9.15% | 9.33% | 10.59% | 3.97% | 4.36% | 12.64% | 17.66% | 5.30% | 2,263.9 ms | 0.44 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-12** | **FCSGG** | CenterNet / RAF | HRNetW32 CenterNet (PyTorch) | **OpenImages V6** | Zero-Shot Transfer | 0.80% | 0.80% | 1.04% | 1.04% | 3.72% | 3.72% | 12.31% | 16.90% | 1.32% | 2,324.0 ms | 0.43 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-13** | **C-SGG** | YOLO12m + CKN (PSG) | ONNX YOLO12m + PyTorch CKN | **PSG** | Hybrid Adaptation | 32.24% | 35.98% | 31.84% | 35.45% | 21.82% | 24.37% | 59.48% | 62.81% | 26.03% | 568.5 ms | 1.76 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-14** | **C-SGG** | YOLOv5l + CKN (VG150) | YOLOv5l + PyTorch CKN | **VG150** | In-Domain | 20.05% | 24.85% | 21.79% | 26.50% | **14.10%** | **15.33%** | 34.19% | 41.37% | 16.56% | **350.3 ms** | **2.85 FPS** | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-15** | **C-SGG** | YOLOv5l + CKN (VG150) | YOLOv5l + PyTorch CKN | **IndoorVG** | Zero-Shot Transfer | 12.15% | 15.22% | 13.82% | 16.81% | 10.81% | 13.03% | 16.56% | 20.88% | 11.44% | 734.0 ms | 1.36 FPS | `VERIFIED_FROM_RAW_CACHE` |
| **RUN-16** | **C-SGG** | YOLOv5l + CKN (OID) | YOLOv5l + PyTorch CKN (OID) | **OpenImages V6** | Hybrid Adaptation | 2.45% | 2.45% | 2.08% | 2.08% | **24.03%** | **24.03%** | 12.81% | 17.08% | 4.44% | 655.0 ms | 1.53 FPS | `VERIFIED_FROM_RAW_CACHE` |

*Verification Note*: All 16 rows hold status `VERIFIED_FROM_RAW_CACHE`. End-to-end CPU re-inference across all 3,200 images was not re-executed during this audit pass (`NOT_EXECUTED_IN_AUDIT`). See [`results/audit_discrepancy_log.md`](results/audit_discrepancy_log.md) for cryptographic provenance and per-run validation records.

### Supplementary Run: CV-SGG on Visual Genome (VG150)
*Context Knowledge Filtering followed by Vision Differentiation Network (VDN) Visual Verification:*
- **Model**: Fast Contextual SGG (CV-SGG)
- **Detector & Backbone**: YOLOv5l detector + CKN candidate filter + ResNet Bottleneck visual verifier (`CKN+VDN+EPBS.pt`)
- **Native Metrics on VG150 (Eligible Images = 196)**:
  - **Image-Avg R@50**: 16.43% | **Image-Avg R@100**: 24.12% | **Image-Avg mR@100**: 13.84%
  - **Detection mAP@50**: 34.19% | **Detection H(AP,AR) [Supp]**: 41.37%
  - **Latency & Throughput**: 4,516.6 ms per image (0.22 FPS on CPU) | **Peak RAM**: 2,271.9 MB
- **Key Takeaway**: While CKN candidate filtering successfully prunes 94.2% of non-interactive pairs (from 2,580 pairs down to ~149 per image), running dense visual crop forward passes through ResNet bottlenecks on CPU incurs a 12.9x latency penalty (4,516 ms vs 350 ms for C-SGG).

---

## 📐 Evaluation Protocols & Ontology View Definitions

To avoid conflation across distinct evaluation populations:

### 1. Recall Metric Formulations
- **Image-Averaged Recall** ($R^{\text{img-avg}}@K$): $\frac{1}{|\mathcal{I}_{\text{eligible}}|} \sum_{i \in \mathcal{I}} \frac{\text{Hits}_i@K}{\text{GT}_i}$. Evaluates the average retrieval rate per eligible image.
- **Conventional Pooled Micro-Recall** ($R^{\text{pooled}}@K$): $\frac{\sum_{i=1}^N \text{Hits}_i@K}{\sum_{i=1}^N \text{GT}_i}$. Evaluates the fraction of total annotated ground-truth triplets retrieved across the entire dataset.
- **Image-Averaged Macro Mean Recall** ($\text{mR}^{\text{img-avg}}@K$): Macro average across active predicate classes of per-image class recalls.
- **Conventional Pooled Macro Mean Recall** ($\text{mR}^{\text{pooled}}@K$): Macro average across active predicate classes of pooled class recalls.

### 2. Ontology Views (Non-Comparable Ground-Truth Populations)
- **Native Ontology View**: Evaluates predictions against the complete, unconstrained vocabulary of the target dataset (OpenPSG: 133 objects / 56 predicates; VG150: 150 objects / 50 predicates; IndoorVG: 84 objects / 31 predicates; OpenImages: 75 objects / 31 predicates).
  - *Eligible Images*: Images with $\ge 1$ ground-truth relation in that native vocabulary.
- **Strict Ontology View** (Visual Genome Exploration): Evaluates strictly against a compact 6-predicate physical contact subset (`touching`, `on`, `holding`, etc.).
  - *Eligible Images*: Images containing $\ge 1$ relation belonging to the strict 6-predicate subset (only 103 of 200 images).
  - *Important*: Because the ground-truth relation count and eligible image population differ drastically between Native and Strict views, **scores from different ontology views cannot be directly compared**.

### 3. OpenImages V6 Dual-Protocol Evaluation
- **Protocol A: Original Standard (866 relations / 200 images)**: Retains all ground-truth relations including unary visual attributes (`is`, `category_id == 27`). Because standard box-pair models have no attribute head, all models score 0 hits on the 720 `is` annotations:
  - C-SGG: **Image-Avg R@50 = 2.45%**, **Pooled R@50 = 2.08%** (18/866 hits), **Image-Avg mR@50 = 24.03%**.
  - REACT: **Image-Avg R@50 = 1.78%**, **Pooled R@50 = 2.08%** (18/866 hits), **Image-Avg mR@50 = 10.76%**.
- **Protocol B: Attribute-Excluded Physical Relations (146 relations / 58 images)**: Excludes predicate 27 `is`:
  - C-SGG: **Image-Avg R@50 = 15.47%**, **Pooled R@50 = 12.33%** (18/146 hits), **Image-Avg mR@50 = 26.04%**.
  - REACT: **Image-Avg R@50 = 12.42%**, **Pooled R@50 = 12.33%** (18/146 hits), **Image-Avg mR@50 = 12.48%**.

---

## 🧠 Architectural Insights & Engineering Takeaways

```mermaid
graph LR
    subgraph Real-Time & Edge Candidates
        CSGG["C-SGG: 350 ms / 2.85 FPS / 935 MB"]
        RPP["REACT++: 548 ms / 1.82 FPS / 789 MB"]
    end
    subgraph Balanced Two-Stage
        REACT["REACT: 701-1224 ms / 0.8-1.4 FPS"]
        FCSGG_M["FCSGG: 1277 ms / 0.78 FPS"]
    end
    subgraph Dense Verification Heavy
        CVSGG["CV-SGG: 4516 ms / 0.22 FPS (ResNet on CPU)"]
    end
```

1. **The Real-Time Frontier**:
   - **REACT++ (ONNX Runtime)** achieves the best Pareto tradeoff on PSG (**37.79% Native R@100**, **548 ms latency**, **789 MB RAM**), delivering a **4.8x memory reduction** over PyTorch.
   - **C-SGG (YOLOv5l + CKN)** achieves the fastest end-to-end CPU throughput (**2.85 FPS**, 350 ms) and the benchmark's highest Native Mean Recall on VG150 (**15.33%**), proving that context reasoning alone (GloVe + geometry) is exceptionally lightweight (4.96 ms for 2,580 pairs).
2. **Detector Precision Dominates SGDet**:
   - In SGDet mode, relationship recall is mathematically bounded by the compound precision of both bounding box endpoints. YOLO12m (mAP@50 = 0.5951) and YOLOv8m (0.5673) significantly outperform single-point CenterNet (0.2253).
3. **Contextual Filtering vs. Dense Crops**:
   - CV-SGG successfully prunes candidate pairs by **94.2%** (from 2,580 pairs down to ~149 pairs per image).
   - However, on CPU, extracting and evaluating 149 individual image crops through ResNet Bottlenecks takes **4,064 ms**, making CV-SGG 12.9x slower than C-SGG.
4. **Ontology Shift**:
   - Visual Genome ground truth is predominantly structural (`has`, `of`, `sitting on`). Mapping down to a 6-predicate physical contact ontology (`strict`) causes sharp recall drop, whereas PSG retains high recall under compact ontologies.

---

## 📁 Repository Organization

```plaintext
minor/
├── README.md                                # Master project documentation & 16-run performance matrix
├── final_report.md                          # Consolidated master benchmark report
├── METHODOLOGY_CROSS_DATASET_BENCHMARKING.md # Cross-dataset compatibility & semantic projection framework
├── evaluation_sets_ai_readable.md           # Dataset audit and metadata specification
├── SGG_Literature_Review_v3.pdf             # Comprehensive literature review
├── evaluation_sets/                         # Frozen 200-image evaluation subsets (Strictly Read-Only)
│   ├── PSG_200.txt / .coco.json / images/   # Panoptic Scene Graph (PSG) benchmark
│   ├── VG150_200.txt / .coco.json / images/ # Visual Genome (VG150) benchmark
│   ├── OIDv6_200.txt / .coco.json / images/ # OpenImages V6 benchmark
│   └── IndoorVG_200.txt / .coco.json / images/ # IndoorVG benchmark
├── experiments/                             # Standalone evaluation runners (16 Model–Dataset Pairs)
│   ├── REACT/                               # run_benchmark_react_{psg,vg150,indoorvg,openimages}.py
│   ├── REACTPP/                             # run_benchmark_reactpp_{psg,vg150,indoorvg,openimages}.py
│   ├── FCSGG/                               # run_benchmark_fcsgg_{psg,vg150,indoorvg,openimages}.py
│   └── FastContextualSGG/                   # run_benchmark_fast_contextual_{psg,vg150,indoorvg,openimages}.py
├── results/                                 # Standardized benchmark deliverables, logs & audit records
│   ├── single_pair_benchmark_matrix.csv     # Complete 16-run tabular matrix (27 columns)
│   ├── single_pair_benchmark_matrix.json    # Machine-readable master benchmark matrix
│   ├── audit_discrepancy_log.md             # Evidence-driven audit log & mathematical derivations
│   ├── reproducibility_audit_records.json   # Cryptographic SHA-256 verification manifest
│   ├── REACT/                               # {dataset}_metrics.csv, .json, and run logs
│   ├── REACTPP/                             # {dataset}_metrics.csv, .json, and run logs
│   ├── FCSGG/                               # {dataset}_metrics.csv, .json, and run logs
│   └── FastContextualSGG/                   # {dataset}_metrics.csv, .json, and run logs
└── model_repos/                             # Source checkouts of evaluated model frameworks
    ├── SGG-Benchmark/                       # Official REACT & SGG-Benchmark codebase
    ├── fcsgg/                               # Official FCSGG codebase with CenterNet / RAF
    └── fast-context-scene-graph-generation/ # Official Fast Contextual SGG codebase
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Nikhil-Verma22/minor_project.git
cd minor_project

# Install required dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime opencv-python pycocotools pyyaml pandas Pillow tabulate tqdm
```

### 2. Model Checkpoint Setup

Pretrained weights are stored outside the Git repository due to file size limits. Place checkpoints into `checkpoints/`:
- **REACT**:
  - PSG / VG150: `checkpoints/REACT/react_PSG/best_model.pth` and `checkpoints/REACT/react_VG150/best_model.pth`
  - IndoorVG: `checkpoints/REACT/react_indoorvg_yolov8m.onnx`
- **REACT++**:
  - PSG / VG150 / IndoorVG / OpenImages: `checkpoints/REACT/react_pp_vg150_yolo12m.onnx` and `checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx`
- **FCSGG**:
  - All Datasets: `checkpoints/FCSGG/HRNetW32-1S.pth`
- **Fast Contextual SGG**:
  - YOLOv5l Detector: `checkpoints/FastContextualSGG/yolol_object_test_28.pt`
  - CKN weights: `checkpoints/FastContextualSGG/` and `model_repos/fast-context-scene-graph-generation/`

### 3. Running Any of the 16 Standalone Benchmarks

Each model–dataset combination has a dedicated standalone runner that handles inference, IoU matching, and metric export:

```bash
# REACT Runs
python experiments/REACT/run_benchmark_react_psg.py
python experiments/REACT/run_benchmark_react_vg150.py
python experiments/REACT/run_benchmark_react_indoorvg.py
python experiments/REACT/run_benchmark_react_openimages.py

# REACT++ Runs
python experiments/REACTPP/run_benchmark_reactpp_psg.py
python experiments/REACTPP/run_benchmark_reactpp_vg150.py
python experiments/REACTPP/run_benchmark_reactpp_indoorvg.py
python experiments/REACTPP/run_benchmark_reactpp_openimages.py

# FCSGG Runs
python experiments/FCSGG/run_benchmark_fcsgg_psg.py
python experiments/FCSGG/run_benchmark_fcsgg_vg150.py
python experiments/FCSGG/run_benchmark_fcsgg_indoorvg.py
python experiments/FCSGG/run_benchmark_fcsgg_openimages.py

# Fast Contextual SGG Runs
python experiments/FastContextualSGG/run_benchmark_fast_contextual_psg.py
python experiments/FastContextualSGG/run_benchmark_fast_contextual_vg150.py
python experiments/FastContextualSGG/run_benchmark_fast_contextual_indoorvg.py
python experiments/FastContextualSGG/run_benchmark_fast_contextual_openimages.py
```

---

## 🔒 Data Integrity & Reproducibility Audit

To guarantee scientific reproducibility and data protection:
- The `evaluation_sets/` directory is treated as **strictly read-only**.
- All **851 files** in `evaluation_sets/` have undergone post-execution cryptographic SHA-256 verification, proving 100% bit-for-bit invariance across all runs.
- The reference SHA-256 digests are preserved in `results/*/eval_sets_sha256_post.json`.

---

## 📄 License & Citations

This benchmark suite builds upon the following official research repositories:
- **REACT / SGG-Benchmark**: [Maelic/SGG-Benchmark](https://github.com/Maelic/SGG-Benchmark)
- **FCSGG**: [liuhengyue/fcsgg](https://github.com/liuhengyue/fcsgg)
- **Fast Contextual SGG**: [moshuilanting/fast-context-scene-graph-generation](https://github.com/moshuilanting/fast-context-scene-graph-generation)

For detailed per-model methodologies and complete per-predicate recall tables, refer to [`final_report.md`](final_report.md).
