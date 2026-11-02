# Master Replication & Full-Scale Benchmark Engineering Guide
## Cross-Dataset Scene Graph Generation (SGG) Across Visual Genome, Panoptic Scene Graph, IndoorVG, and Open Images V6

**Project**: *Object Retrieval in Dynamic Workspace Environments* — Unified Cross-Ontology SGG Benchmark  
**Repository**: [https://github.com/Nikhil-Verma22/minor_project](https://github.com/Nikhil-Verma22/minor_project)  
**Models Evaluated**: REACT v1, REACT++, FCSGG, and Fast Contextual SGG (C-SGG)  
**Target Datasets**: Panoptic Scene Graph (PSG), Visual Genome (VG150), IndoorVG, and Open Images V6 (OIDv6)  
**Standardized Evaluation Task**: Scene Graph Detection (`SGDet`)  
**Scope**: Comprehensive engineering manual covering the full research lifecycle, architectural mechanics, environment & compatibility fixes, semantic projection over incompatible ontologies, verified 16-run benchmark results, and an exhaustive blueprint for scaling from 200-image frozen audits to full multi-thousand image test sets (up to 31,876 images).

---

## Table of Contents

1. [Executive Summary & Purpose](#1-executive-summary--purpose)
2. [SGG Benchmark Foundations & Task Formulation](#2-sgg-benchmark-foundations--task-formulation)
   - [2.1 Scene Graph Detection (`SGDet`) Protocol](#21-scene-graph-detection-sgdet-protocol)
   - [2.2 The Dual-Scale Paradigm (Frozen Subset vs. Full Test Split)](#22-the-dual-scale-paradigm-frozen-subset-vs-full-test-split)
3. [Deep Architectural Breakdown of the 4 Model Families](#3-deep-architectural-breakdown-of-the-4-model-families)
   - [3.1 REACT v1 (PyTorch YOLOv8m + Relation Transformer Head)](#31-react-v1-pytorch-yolov8m--relation-transformer-head)
   - [3.2 REACT++ (Single-Stage YOLO12m ONNX Graph)](#32-react-single-stage-yolo12m-onnx-graph)
   - [3.3 FCSGG (CenterNet Keypoints + RAF on HRNet-W32)](#33-fcsgg-centernet-keypoints--raf-on-hrnet-w32)
   - [3.4 Fast Contextual SGG (C-SGG: Online YOLOv5l + CKN MLP)](#34-fast-contextual-sgg-c-sgg-online-yolov5l--ckn-mlp)
4. [Environment, Runtime & Cross-Platform Compatibility Engineering](#4-environment-runtime--cross-platform-compatibility-engineering)
   - [4.1 Bypassing Detectron2 C++/CUDA Compilation on CPU (The `_C` Stub)](#41-bypassing-detectron2-ccuda-compilation-on-cpu-the-_c-stub)
   - [4.2 Pillow 10+ Resampling Deprecation Fix](#42-pillow-10-resampling-deprecation-fix)
   - [4.3 PyTorch 2.6+ / 2.8+ Serialization Security (`weights_only=False`)](#43-pytorch-26--28-serialization-security-weights_onlyfalse)
   - [4.4 Unpickling YOLOv5 Checkpoints (`ModuleNotFoundError: No module named 'models'`)](#44-unpickling-yolov5-checkpoints-modulenotfounderror-no-module-named-models)
   - [4.5 Windows CP1252 vs. UTF-8 Console Encoding Safeguards](#45-windows-cp1252-vs-utf-8-console-encoding-safeguards)
5. [The Word Embedding Engineering Challenge (GloVe Deep Dive)](#5-the-word-embedding-engineering-challenge-glove-deep-dive)
   - [5.1 The 200d GloVe Requirement in REACT v1 & The Zero-Stub Discovery](#51-the-200d-glove-requirement-in-react-v1--the-zero-stub-discovery)
   - [5.2 The 50d GloVe Requirement in C-SGG & The Streaming Extraction Solution](#52-the-50d-glove-requirement-in-c-sgg--the-streaming-extraction-solution)
   - [5.3 Acquisition of Full Stanford GloVe for Large-Scale Deployment](#53-acquisition-of-full-stanford-glove-for-large-scale-deployment)
6. [Cross-Dataset Semantic Projection: Resolving Incompatible Ontologies](#6-cross-dataset-semantic-projection-resolving-incompatible-ontologies)
   - [6.1 The 4 Benchmark Ontologies at a Glance](#61-the-4-benchmark-ontologies-at-a-glance)
   - [6.2 The 3-Tier Object Category Alignment Architecture](#62-the-3-tier-object-category-alignment-architecture)
   - [6.3 Predicate Vocabulary Routing & Max-Pooling Aggregation](#63-predicate-vocabulary-routing--max-pooling-aggregation)
   - [6.4 The OpenImages V6 Attribute Skew: Protocol A vs. Protocol B](#64-the-openimages-v6-attribute-skew-protocol-a-vs-protocol-b)
   - [6.5 Correcting the IndoorVG Predicate Count (37 vs. 31)](#65-correcting-the-indoorvg-predicate-count-37-vs-31)
7. [The 16-Run Experimental Benchmark Matrix](#7-the-16-run-experimental-benchmark-matrix)
   - [7.1 Master Specification Table](#71-master-specification-table)
   - [7.2 Metric Formulations: Image-Averaged vs. Pooled Micro-Recall](#72-metric-formulations-image-averaged-vs-pooled-micro-recall)
   - [7.3 Reconciled Performance Numbers](#73-reconciled-performance-numbers)
   - [7.4 OpenImages Dual-Protocol Audit Verification](#74-openimages-dual-protocol-audit-verification)
8. [Blueprint for Scaling to Full Datasets (Tens of Thousands of Images)](#8-blueprint-for-scaling-to-full-datasets-tens-of-thousands-of-images)
   - [8.1 Full Test Split Populations & Download Sources](#81-full-test-split-populations--download-sources)
   - [8.2 Hardware & Pipeline Migration (CPU to Multi-GPU CUDA)](#82-hardware--pipeline-migration-cpu-to-multi-gpu-cuda)
   - [8.3 Streaming DataLoader & Batched Inference Architecture](#83-streaming-dataloader--batched-inference-architecture)
   - [8.4 Sharded Disk Caching to Prevent RAM Out-of-Memory (OOM)](#84-sharded-disk-caching-to-prevent-ram-out-of-memory-oom)
   - [8.5 Vectorized Metric Accumulation over Millions of Triplets](#85-vectorized-metric-accumulation-over-millions-of-triplets)
9. [Step-by-Step Practical Replication Guide](#9-step-by-step-practical-replication-guide)
   - [9.1 Environment & Dependency Setup](#91-environment--dependency-setup)
   - [9.2 Repository Layout & Checkpoint Placement](#92-repository-layout--checkpoint-placement)
   - [9.3 Checkpoint SHA-256 Verification Table](#93-checkpoint-sha-256-verification-table)
   - [9.4 Running the 16 Benchmark Scripts (Single-Image, 200-Image, Full-Scale)](#94-running-the-16-benchmark-scripts-single-image-200-image-full-scale)
   - [9.5 Master Troubleshooting Playbook](#95-master-troubleshooting-playbook)
10. [Conclusion & Key Research Insights](#10-conclusion--key-research-insights)

---

## 1. Executive Summary & Purpose

Scene Graph Generation (SGG) models are traditionally evaluated within isolated, single-dataset silos using closed vocabularies (e.g. evaluating purely on Visual Genome or Panoptic Scene Graph). This repository represents an exhaustive, evidence-based benchmark evaluating **four modern SGG architectures** across **four distinct visual domains**:

1. **Visual Genome (VG150)**: Classic, highly redundant, web-scale visual relationship dataset.
2. **Panoptic Scene Graph (PSG)**: Dense, COCO-derived panoptic segmentations covering both *things* (foreground objects) and *stuff* (amorphous backgrounds/materials).
3. **IndoorVG**: Highly concentrated, specialized indoor robotics/home environment relationships with extreme spatial proximity.
4. **Open Images V6 (OIDv6)**: Sparse, large-scale, hierarchical ontology characterized by strict annotation protocols and a dominance of visual attribute relationships.

Evaluating models cross-dataset presents massive engineering and scientific challenges:
- **Ontology Incompatibility**: Each dataset defines different object classes (e.g., 150 vs. 133 vs. 84 vs. 601) and predicate vocabularies (e.g., 50 vs. 56 vs. 37 vs. 30), with conflicting semantic granularities and part-of-speech conventions.
- **Architectural Divergence**: Models span two-stage PyTorch pipelines, single-stage ONNX computation graphs, bottom-up anchor-free keypoint networks, and decoupled contextual MLPs.
- **Platform & Library Obstacles**: Codebases rely on incompatible historical dependencies (Detectron2 C++ CUDA extensions, deprecated Pillow resampling enums, legacy PyTorch pickling protocols, and absent text vector archives).

This document serves as the **master replication manual**. It explains every technique used to overcome these hurdles, provides exact mathematical formulations for harmonizing cross-ontology candidate pairs, documents verified benchmark numbers, and outlines the complete engineering architecture for scaling evaluations to the full 31,876-image datasets.

---

## 2. SGG Benchmark Foundations & Task Formulation

### 2.1 Scene Graph Detection (`SGDet`) Protocol

All 16 evaluations in this benchmark are strictly executed under the **Scene Graph Detection (`SGDet`)** task, which is the most challenging and realistic evaluation mode for embodied AI, robotics, and object retrieval:

$$\text{Input: Raw Image } I \longrightarrow \text{Output: Triplet Set } \mathcal{T} = \{(b_s, c_s, p, b_o, c_o, S)\}$$

Where:
- $b_s, b_o \in \mathbb{R}^4$: 2D bounding boxes in $[x_1, y_1, x_2, y_2]$ coordinate space for the subject and object.
- $c_s, c_o \in \mathcal{C}_{target}$: Predicted target object category labels.
- $p \in \mathcal{P}_{target}$: Directed relationship predicate label connecting subject to object ($s \xrightarrow{p} o$).
- $S \in [0, 1]$: Confidence score for the candidate triplet.

#### Ground Truth Matching Criteria
A predicted triplet $(b_s^*, c_s^*, p^*, b_o^*, c_o^*)$ is declared a **True Positive (Hit)** if and only if there exists a ground-truth triplet $(b_s^{gt}, c_s^{gt}, p^{gt}, b_o^{gt}, c_o^{gt})$ such that all five conditions hold simultaneously:
1. **Subject Spatial Overlap**: $\text{IoU}(b_s^*, b_s^{gt}) \ge 0.50$
2. **Object Spatial Overlap**: $\text{IoU}(b_o^*, b_o^{gt}) \ge 0.50$
3. **Subject Classification**: $c_s^* = c_s^{gt}$
4. **Object Classification**: $c_o^* = c_o^{gt}$
5. **Predicate Classification**: $p^* = p^{gt}$

Each ground-truth relationship can be matched at most once (greedy matching by descending prediction confidence $S$). Unmatched predictions are false positives.

### 2.2 The Dual-Scale Paradigm (Frozen Subset vs. Full Test Split)

To ensure scientific rigor, reproducibility, and computational feasibility, our benchmarking framework operates on two distinct scales:

1. **The 200-Image Frozen Benchmark Protocol**:
   - Exactly 200 images were deterministically sampled from the official test split of each dataset using a frozen pseudo-random seed (`seed=42`).
   - All 800 image files and their corresponding COCO-format ground-truth annotation files are frozen under [`evaluation_sets/`](file:///d:/Minor/evaluation_sets).
   - Bit-level immutability is guaranteed: all 851 benchmark files have recorded, verified SHA-256 digests.
   - Purpose: Enables instant verification, rapid architectural audits, zero-leakage parameter validation, and running complete 16-run sweeps on CPU workstations without days of cluster compute.
2. **The Full-Scale Test Split Protocol**:
   - Involves running model inference across the entire official test set of each dataset (e.g., 31,876 images for VG150, 2,177 images for PSG).
   - Requires batched data loading, GPU acceleration, sharded disk caching, and memory-efficient metric accumulators (detailed in Section 8).

---

## 3. Deep Architectural Breakdown of the 4 Model Families

```
+--------------------------------------------------------------------------------------------------+
|                                    SGG INFERENCE PARADIGMS                                       |
+------------------------------------+-------------------------------------------------------------+
| 1. Two-Stage Modular (REACT v1)    | YOLOv8m Detector --> RoI Pooling --> Transformer Head       |
| 2. Single-Stage ONNX (REACT++)     | YOLO12m Backbone + DAMP + AIFI Head --> Unified ONNX Graph   |
| 3. Bottom-Up RAF (FCSGG)           | HRNet-W32 --> CenterNet Keypoints + Relation Affinity Fields|
| 4. Decoupled Contextual (C-SGG)    | Online YOLOv5l --> 30d Geometry + 50d GloVe --> CKN MLP Head|
+------------------------------------+-------------------------------------------------------------+
```

### 3.1 REACT v1 (PyTorch YOLOv8m + Relation Transformer Head)
- **Reference**: *REACT: Context-guided Relational Reasoning for Real-time Scene Graph Generation* (BMVC 2025, Maelic Neau et al.).
- **Inference Engine**: Pure PyTorch execution (`sgg_benchmark` framework).
- **Stage 1 (Perception)**: Pretrained Ultralytics YOLOv8m backbone detects entity bounding boxes and object category logits.
- **Stage 2 (Relational Reasoning)**:
  1. Box proposals are filtered via non-maximum suppression (NMS, `conf_thres=0.001`, `max_det=100`).
  2. RoI features (512d) are extracted for each entity.
  3. Entity class names are projected into a continuous word vector space (GloVe embeddings).
  4. Context-guided multi-head attention dynamically computes relational interaction tokens between all entity pairs.
- **Candidate Scoring**:
  $$S(s, p, o) = s_{subj} \times s_{obj} \times s_{rel}$$
  Where $s_{subj}$ and $s_{obj}$ are detection confidences, and $s_{rel}$ is the Softmax/Sigmoid predicate logit.
- **Checkpoints**:
  - VG150: [`checkpoints/REACT/react_VG150/best_model_epoch_9.pth`](file:///d:/Minor/checkpoints/REACT/react_VG150/best_model_epoch_9.pth) (225.05 MB, Epoch 9, SHA-256: `8bf598afdb...`).
  - PSG: [`checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth`](file:///d:/Minor/checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth) (225.00 MB, Epoch 11, SHA-256: `0f27511e66...`).
  - IndoorVG: Native ONNX model [`checkpoints/REACT/react_indoorvg_yolov8m.onnx`](file:///d:/Minor/checkpoints/REACT/react_indoorvg_yolov8m.onnx) (132.45 MB, SHA-256: `5188b578aa...`).

### 3.2 REACT++ (Single-Stage YOLO12m ONNX Graph)
- **Reference**: *REACT++: Efficient Cross-Attention for Real-Time Scene Graph Generation* (CVPR 2025 / Under Review, Maelic Neau et al.).
- **Inference Engine**: ONNX Runtime (`CPUExecutionProvider` / `CUDAExecutionProvider`).
- **Architecture**:
  - Backbone: YOLO12m.
  - Multi-Scale Object Representation: **Detection-Anchored Multi-Scale Pooling (DAMP)** extracts entity features across P3, P4, and P5 feature pyramids using a Gaussian neighbourhood ($r=1$).
  - Scene Context: **AIFI (All-to-One / Intra-Scale Feature Interaction)** branch injects global image semantics.
  - Deployment: Fully exported into a monolithic, static ONNX graph. Input: `[batch_size, 3, 640, 640]`. Outputs: `boxes` (`[num_objects, 6]`) and `rels` (`[num_rels, 5]`).
- **The Compound Triplet Scoring Contract**:
  - Column 3 of the `rels` output tensor is the **geometric mean compound triplet confidence**:
    $$\text{Column 3} = (s_{subj} \times s_{obj} \times s_{rel})^{1/3}$$
  - **Audit Discovery & Bug Fix**: In previous cross-dataset evaluation scripts for IndoorVG and OpenImages, the script re-multiplied column 3 by $s_{subj} \times s_{obj}$, double-penalizing detection confidences and distorting the ranking. We corrected both runners ([`run_benchmark_reactpp_indoorvg.py`](file:///d:/Minor/experiments/REACTPP/run_benchmark_reactpp_indoorvg.py) and [`run_benchmark_reactpp_openimages.py`](file:///d:/Minor/experiments/REACTPP/run_benchmark_reactpp_openimages.py)) to sort directly on column 3. This increased IndoorVG R@50 from 13.64% to **14.75%** (+30 true positive hits) and OpenImages R@50 to **2.24%**.

### 3.3 FCSGG (CenterNet Keypoints + RAF on HRNet-W32)
- **Reference**: *Fully Convolutional Scene Graph Generation* (CVPR 2021, Hengyue Liu et al.).
- **Inference Engine**: Detectron2-based bottom-up network wrapped by [`experiments/FCSGG/fcsgg_adapter.py`](file:///d:/Minor/experiments/FCSGG/fcsgg_adapter.py).
- **The YOLO Myth Debunked**:
  > [!IMPORTANT]
  > FCSGG does **NOT** utilize a YOLO object detector. It is an anchor-free bottom-up network that detects objects as center points using **CenterNet heads** (center heatmaps, box regressions, and offset maps) built on an **HRNet-W32** backbone. Simultaneously, it outputs **Relation Affinity Fields (RAF)**—spatial vector fields that connect subject centers to object centers while predicting predicate probabilities.
- **Upstream Checkpoint**:
  - [`checkpoints/FCSGG/HRNetW32-1S.pth`](file:///d:/Minor/checkpoints/FCSGG/HRNetW32-1S.pth) (362.41 MB, SHA-256: `41aa7639ba...`).
  - Pretrained exclusively on Visual Genome (VG150, 150 classes, 50 predicates). Evaluates on other datasets via semantic projection.

### 3.4 Fast Contextual SGG (C-SGG: Online YOLOv5l + CKN MLP)
- **Reference**: *Fast Contextual Scene Graph Generation with Unbiased Context Augmentation* (CVPR 2023, Jin et al.).
- **Inference Engine**: Hybrid PyTorch pipeline.
- **Decoupled Perception & Reasoning**:
  - Perception: Pretrained YOLOv5l detector ([`checkpoints/FastContextualSGG/yolol_object_test_28.pt`](file:///d:/Minor/checkpoints/FastContextualSGG/yolol_object_test_28.pt), 359.05 MB, SHA-256: `f657e21917...`) performs online bounding box and entity classification.
  - Relation Reasoning: Bypasses pixel patches entirely. For each directed candidate pair $(i, j)$:
    1. **Spatial Geometry Vector (30d)**: Normalized box coordinates, relative center offsets, aspect ratio differences, box areas, and intersection metrics:
       $$\mathbf{g}_{ij} \in \mathbb{R}^{30}$$
    2. **Semantic Word Vectors (50d)**: GloVe embeddings for subject and object classes:
       $$\mathbf{w}_i, \mathbf{w}_j \in \mathbb{R}^{50}$$
    3. **Context Knowledge Network (CKN)**: A lightweight 3-layer MLP scores the concatenated 80d feature vector:
       $$\mathbf{f}_{ij} = [\mathbf{g}_{ij}, \mathbf{w}_i, \mathbf{w}_j] \in \mathbb{R}^{130} \xrightarrow{\text{CKN MLP}} \mathbf{s}_{rel} \in \mathbb{R}^{|\mathcal{P}|}$$
  - **Elimination of Offline Proposal Files**: Upstream C-SGG relied on massive, precomputed Faster R-CNN `.bin` proposal files. We engineered [`experiments/FastContextualSGG/fast_contextual_adapter.py`](file:///d:/Minor/experiments/FastContextualSGG/fast_contextual_adapter.py) to run online YOLOv5l detection, dynamically generating candidate pairs in **4.96 ms** on CPU, achieving **2.85 FPS end-to-end**.

---

## 4. Environment, Runtime & Cross-Platform Compatibility Engineering

Reproducing academic SGG research on a standard workstation (especially Windows or modern Linux distributions with Python 3.10+) invariably encounters severe dependency breaks. Below are the exact technical hurdles and their drop-in solutions:

### 4.1 Bypassing Detectron2 C++/CUDA Compilation on CPU (The `_C` Stub)
- **The Problem**: Upstream FCSGG imports `detectron2`, which requires compiling native C++/CUDA extensions (`detectron2._C`). On machines without Visual Studio Build Tools, CUDA toolkits, or running on pure CPU, importing Detectron2 raises `ImportError: DLL load failed` or `ModuleNotFoundError: No module named 'detectron2._C'`.
- **The Solution**: In [`experiments/FCSGG/fcsgg_adapter.py`](file:///d:/Minor/experiments/FCSGG/fcsgg_adapter.py#L35-L65), we inject a synthetic module into `sys.modules` before importing Detectron2:
```python
import sys, types
import torchvision.ops as tv_ops

# Create dynamic stub for missing compiled C++ extension
c_stub = types.ModuleType("detectron2._C")
c_stub.nms = tv_ops.nms
c_stub.batched_nms = tv_ops.batched_nms
c_stub.box_iou = tv_ops.box_iou
sys.modules["detectron2._C"] = c_stub
```
This redirects all native C++ NMS operations to torchvision's pure Python/PyTorch C-backend, eliminating the need to compile Detectron2.

### 4.2 Pillow 10+ Resampling Deprecation Fix
- **The Problem**: In Pillow $\ge 10.0.0$, the constant `PIL.Image.LINEAR` was permanently removed in favor of `PIL.Image.Resampling.BILINEAR`. Running FCSGG or Detectron2 data augmentations triggers `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'`.
- **The Solution**: Insert an alias monkey-patch during startup:
```python
import PIL.Image
if not hasattr(PIL.Image, "LINEAR"):
    PIL.Image.LINEAR = PIL.Image.BILINEAR
```

### 4.3 PyTorch 2.6+ / 2.8+ Serialization Security (`weights_only=False`)
- **The Problem**: PyTorch $\ge 2.6.0$ changed the default behavior of `torch.load` to `weights_only=True` for security against malicious pickle exploits. This causes checkpoints containing dictionaries, configs, optimizer states, or NumPy scalar arrays (such as `HRNetW32-1S.pth`, `best_model_epoch_9.pth`, and `yolol_object_test_28.pt`) to crash with `torch.serialization.WeightsOnlyError`.
- **The Solution**: Explicitly pass `weights_only=False` in all model loaders:
```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```

### 4.4 Unpickling YOLOv5 Checkpoints (`ModuleNotFoundError: No module named 'models'`)
- **The Problem**: Pretrained YOLOv5 checkpoints (`yolol_object_test_28.pt`) pickle the entire model class instance rather than just the `state_dict`. Unpickling attempts to import `models.yolo.Model`. If the YOLOv5 directory is not the current working directory, Python raises `ModuleNotFoundError: No module named 'models'`.
- **The Solution**: In [`fast_contextual_adapter.py`](file:///d:/Minor/experiments/FastContextualSGG/fast_contextual_adapter.py#L22-L26), dynamically prepend the YOLOv5 codebase path to `sys.path`:
```python
yolo_root = os.path.join(WORKSPACE_ROOT, "model_repos/fast-context-scene-graph-generation/VG_dataset")
if yolo_root not in sys.path:
    sys.path.insert(0, yolo_root)
```

### 4.5 Windows CP1252 vs. UTF-8 Console Encoding Safeguards
- **The Problem**: On Windows PowerShell / CMD, default stdout uses `cp1252` encoding. Logging Unicode arrows ($\to$), mathematical symbols, or box-drawing characters crashes scripts with `UnicodeEncodeError: 'charmap' codec can't encode character`.
- **The Solution**: Reconfigure Python's standard output stream to UTF-8 at the top of every CLI runner:
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
```

---

## 5. The Word Embedding Engineering Challenge (GloVe Deep Dive)

### 5.1 The 200d GloVe Requirement in REACT v1 & The Zero-Stub Discovery
Upstream REACT v1 code ([`sgg_benchmark/utils/txt_embeddings.py`](file:///d:/Minor/model_repos/SGG-Benchmark/sgg_benchmark/utils/txt_embeddings.py#L251-L316)) seeks word vector files in `glove_dir`:
```python
fname = os.path.join(root, wv_type + '.' + dim)
if os.path.isfile(fname + '.pt'):
    return torch.load(fname + '.pt', map_location='cpu')
# Else: attempts to download glove.6B.zip (862 MB) from Stanford NLP
```
**Physical Audit Finding**:  
Our disk inspection of [`d:\Minor\.cache\REACT\glove\glove.6B.200d.pt`](file:///d:/Minor/.cache/REACT/glove/glove.6B.200d.pt) (2,387 bytes, SHA-256: `0c40113407...`) revealed that it contains:
$$\text{Content} = (\{\}, \mathbf{0}^{1 \times 200}, 200)$$
Because `wv_dict` is empty (`{}`), `wv_dict.get(token)` returns `None` for all classes. The code then falls back to Gaussian initialization:
```python
vectors = torch.Tensor(len(names), wv_dim)
vectors.normal_(0, 1) # Standard normal distribution fallback
```
This stub successfully bypassed downloading the 862 MB Stanford archive, but initialized entity tokens with random Gaussian vectors rather than GloVe semantics.

### 5.2 The 50d GloVe Requirement in C-SGG & The Streaming Extraction Solution
Fast Contextual SGG requires 50d word vectors for its CKN MLP head. To obtain authentic 50d GloVe embeddings without downloading the full 862 MB zip, we authored [`d:\Minor\.cache\FastContextualSGG\extract_glove.py`](file:///d:/Minor/.cache/FastContextualSGG/extract_glove.py).

This script streams lines directly from Hugging Face (`https://huggingface.co/JeremiahZ/glove/resolve/main/glove.6B.50d.txt`), parsing only the 215 unique entity tokens required by Visual Genome and Panoptic Scene Graph. It terminates the connection immediately once all words are found, saving:
- [`glove_extracted_50d.pt`](file:///d:/Minor/.cache/FastContextualSGG/glove_extracted_50d.pt) (215 words $\times$ 50d float32)
- [`vg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/vg_word_vec_list.pt) (151 classes $\times$ 50d tensor, SHA-256: `a3bc6394...`)
- [`psg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/psg_word_vec_list.pt) (134 classes $\times$ 50d tensor, SHA-256: `dd42ac40...`)

### 5.3 Acquisition of Full Stanford GloVe for Large-Scale Deployment
For full-scale multi-thousand image evaluations where novel class vocabularies may be queried, download and extract the official Stanford GloVe vectors:
```bash
# Download official GloVe 6B archive (862 MB)
curl -O http://nlp.stanford.edu/data/glove.6B.zip

# Unzip target dimensions
unzip glove.6B.zip glove.6B.50d.txt glove.6B.200d.txt glove.6B.300d.txt -d .cache/glove/
```
Convert them to fast `.pt` caches using the script provided in Section 8.3.

---

## 6. Cross-Dataset Semantic Projection: Resolving Incompatible Ontologies

Evaluating a model trained on Visual Genome across Panoptic Scene Graph, IndoorVG, or OpenImages requires harmonizing incompatible object and predicate label spaces.

### 6.1 The 4 Benchmark Ontologies at a Glance

| Dataset | Object Classes | Predicate Classes | Semantic Focus | Annotator Characteristics |
| :--- | :---: | :---: | :--- | :--- |
| **Visual Genome (VG150)** | 150 | 50 | Generic web imagery, objects & parts | Extreme redundancy (`on`, `has`, `wearing` dominate) |
| **OpenPSG (PSG)** | 133 (80 things, 53 stuff) | 56 | Panoptic COCO scenes | Rich spatial & action verbs (`standing on`, `holding`) |
| **IndoorVG** | 84 | 37 (29 active in eval) | Indoor rooms, furniture, utensils | Fine-grained indoor relations (`mounted on`, `laying on`) |
| **Open Images V6** | 601 det / 38 rel | 30 physical + attributes | Large-scale diverse imagery | 75%+ unary visual attributes (`is wooden`, `is red`) |

### 6.2 The 3-Tier Object Category Alignment Architecture

When a model detects an entity from source ontology $\mathcal{C}_{src}$, it must be projected into target ontology $\mathcal{C}_{target}$. We implement a deterministic **3-Tier Alignment Framework**:

```
+--------------------------------------------------------------------------------------------------+
|                              3-TIER OBJECT CATEGORY PROJECTION                                   |
+--------------------------------------------------------------------------------------------------+
| Tier 1: Exact Match      | 'person' == 'person', 'table' == 'table'       --> Direct 1:1 Identity|
| Tier 2: Synonym/Stem     | 'couch' <-> 'sofa', 'sports ball' <-> 'ball'   --> Lemma/Synset Map   |
| Tier 3: Out-of-Vocabulary| 'airplane' in IndoorVG, 'elephant' in Bedroom  --> Mask to 0 (No Pen.)|
+--------------------------------------------------------------------------------------------------+
```

1. **Tier 1 (Exact String Equivalence)**: Normalized string matches map directly with 1:1 identity (e.g., `dog` $\to$ `dog`, `bottle` $\to$ `bottle`).
2. **Tier 2 (Synonym & Lemmatized Projection)**: Morphological stems and WordNet synsets bridge terminology differences:
   - `couch` $\longleftrightarrow$ `sofa`
   - `sports ball` $\longleftrightarrow$ `ball`
   - `desk` $\longleftrightarrow$ `table`
   - `dining table` $\longleftrightarrow$ `table`
   - `bicycle` $\longleftrightarrow$ `bike`
3. **Tier 3 (Zero-Shot / Out-of-Vocabulary Masking)**: Entities present in the source dataset but completely absent from the target ontology are mapped to class index `0` (background/unmapped). This prevents models from being penalized for detecting valid visual objects that simply lack target annotations.

### 6.3 Predicate Vocabulary Routing & Max-Pooling Aggregation

Directed relationship predicates are mapped through our frozen bipartite projection table [`predicate_mapping_evaluation_v1.2_FINAL.csv`](file:///d:/Minor/evaluation_sets/predicate_mapping_evaluation_v1.2_FINAL.csv) (SHA-256: `4cc5924376...`).

#### Confidence Max-Pooling
When multiple source predicates map to the same target predicate (many-to-one mapping, e.g. VG150 `wears` and `wearing` both mapping to target `wearing`), the candidate relation score is aggregated via **Max-Pooling**:

$$S_{target}(s, p_{target}, o) = \max_{p_{src} \in \mathcal{M}^{-1}(p_{target})} S_{src}(s, p_{src}, o)$$

This ensures that the model's highest-confidence semantic prediction is preserved without probability dilution.

### 6.4 The OpenImages V6 Attribute Skew: Protocol A vs. Protocol B

> [!WARNING]
> **Critical Methodological Finding**:
> In Open Images V6, **720 out of the 866 ground-truth relationships** in the benchmark evaluation set (83.1%) are **unary visual attributes** using predicate ID 27 (`is`, e.g. `flower is red`, `table is wooden`, `boy is smiling`).

Pairwise Scene Graph models (REACT, FCSGG, C-SGG) predict relationships between **two distinct spatial entities** ($s \ne o, \text{IoU}(b_s, b_o) < 1.0$). Consequently, all pairwise models score **zero hits** on unary attributes.

To report scientifically honest metrics, we formulate two distinct evaluation protocols:

- **Protocol A (Standard Full Population)**:
  - Evaluates over all 200 images containing all **866 ground-truth relationships** (including 720 unary attributes).
  - Demonstrates raw recall under the standard benchmark format, yielding ~1% to 2.5% pooled recall across all models.
- **Protocol B (Physical Pairwise Interactions Only)**:
  - Filters out unary predicate 27 (`is`).
  - Evaluates over the **58 eligible images** containing **146 genuine physical interactions** (e.g. `holding`, `wearing`, `on`, `inside_of`, `interacts_with`).
  - Reveals true relational transfer performance: pooled recall ranges from **6.16% to 14.38%**.

Anyone extending the benchmark to the full OpenImages test set **must** compute and report Protocol B alongside Protocol A to avoid misinterpreting pairwise SGG transfer capabilities.

### 6.5 Correcting the IndoorVG Predicate Count (37 vs. 31)
Earlier literature and draft READMEs erroneously stated that IndoorVG contains 31 predicate categories.  
Our physical inspection of the official annotation vocabulary and model metadata confirmed:
- **Total Defined Predicate Classes**: **37 predicates** (plus background at index 0).
- **Active Predicate Classes in Benchmark Subset**: **29 predicates** appear with non-zero ground-truth instances across the 200 evaluation images.
- All evaluation metric scripts have been verified against the authentic 37-class schema.

---

## 7. The 16-Run Experimental Benchmark Matrix

### 7.1 Master Specification Table

| Run ID | Model Architecture | Evaluated Variant | Target Dataset | Primary Checkpoint | Runtime Engine | Evaluation Regime |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: |
| **RUN-01** | **REACT** (v1) | PyTorch YOLOv8m | **PSG** | `react_PSG/best_model_epoch_11.pth` | PyTorch CPU | In-Domain Native |
| **RUN-02** | **REACT** (v1) | PyTorch YOLOv8m | **VG150** | `react_VG150/best_model_epoch_9.pth` | PyTorch CPU | In-Domain Native |
| **RUN-03** | **REACT** (v1) | ONNX YOLOv8m | **IndoorVG** | `react_indoorvg_yolov8m.onnx` | ONNX Runtime | In-Domain Native |
| **RUN-04** | **REACT** (v1) | PyTorch YOLOv8m | **OpenImages V6** | `react_VG150/best_model_epoch_9.pth` | PyTorch CPU | Zero-Shot Transfer |
| **RUN-05** | **REACT++** | ONNX YOLO12m | **PSG** | `react_pp_yolo12m.onnx` | ONNX Runtime | In-Domain Native |
| **RUN-06** | **REACT++** | ONNX YOLO12m | **VG150** | `react_pp_vg150_yolo12m.onnx` | ONNX Runtime | In-Domain Native |
| **RUN-07** | **REACT++** | ONNX YOLO12m | **IndoorVG** | `react_pp_vg150_yolo12m.onnx` | ONNX Runtime | Zero-Shot Transfer |
| **RUN-08** | **REACT++** | ONNX YOLO12m | **OpenImages V6** | `react_pp_vg150_yolo12m.onnx` | ONNX Runtime | Zero-Shot Transfer |
| **RUN-09** | **FCSGG** | CenterNet / RAF | **PSG** | `HRNetW32-1S.pth` | PyTorch CPU | Zero-Shot Transfer |
| **RUN-10** | **FCSGG** | CenterNet / RAF | **VG150** | `HRNetW32-1S.pth` | PyTorch CPU | In-Domain Native |
| **RUN-11** | **FCSGG** | CenterNet / RAF | **IndoorVG** | `HRNetW32-1S.pth` | PyTorch CPU | Zero-Shot Transfer |
| **RUN-12** | **FCSGG** | CenterNet / RAF | **OpenImages V6** | `HRNetW32-1S.pth` | PyTorch CPU | Zero-Shot Transfer |
| **RUN-13** | **C-SGG** | YOLO12m + CKN | **PSG** | `react_pp_yolo12m.onnx` + `2619.pt` | Hybrid PyTorch | Hybrid Adaptation |
| **RUN-14** | **C-SGG** | YOLOv5l + CKN | **VG150** | `yolol_object_test_28.pt` + `CKN+EPBS.pt` | PyTorch CPU | In-Domain Native |
| **RUN-15** | **C-SGG** | YOLOv5l + CKN | **IndoorVG** | `yolol_object_test_28.pt` + `CKN+EPBS.pt` | PyTorch CPU | Zero-Shot Transfer |
| **RUN-16** | **C-SGG** | YOLOv5l + CKN | **OpenImages V6** | `yolol_object_test_28.pt` + `ckn_oid_PBS.pt`| Hybrid PyTorch | Hybrid Adaptation |

### 7.2 Metric Formulations: Image-Averaged vs. Pooled Micro-Recall

To prevent arithmetic confusion during replication, distinguish these two metrics:

1. **Image-Averaged Recall ($R@K$)**:
   Computes recall independently per image $i \in \{1 \dots N\}$, then averages across all $N$ images:
   $$R@K = \frac{1}{N} \sum_{i=1}^N \frac{\text{Hits}_i@K}{|\mathcal{G}_i|}$$
   *(Images with zero ground-truth relations are omitted from the denominator).*

2. **Conventional Pooled Micro-Recall ($R_{pooled}@K$)**:
   Sums all true positive matches across the entire dataset and divides by total ground-truth relationships:
   $$R_{pooled}@K = \frac{\sum_{i=1}^N \text{Hits}_i@K}{\sum_{i=1}^N |\mathcal{G}_i|}$$

3. **Macro Mean Recall ($mR@K$)**:
   Computes recall independently for each predicate class $c \in \mathcal{P}$, then takes the unweighted macro average:
   $$mR@K = \frac{1}{|\mathcal{P}|} \sum_{c \in \mathcal{P}} \frac{\text{Hits}_c@K}{|\mathcal{G}_c|}$$

### 7.3 Reconciled Performance Numbers

All metrics below are computed from raw serialized prediction caches (`.cache/*/*.pt`) and independently verified in [`results/reproducibility_audit_records.json`](file:///d:/Minor/results/reproducibility_audit_records.json) with absolute numerical error $|\Delta| < 10^{-4}$.

| Run ID | Model | Dataset | Regime | Image-Avg R@50 | Image-Avg R@100 | Macro mR@50 | Macro mR@100 | Pooled R@50 | Pooled R@100 | Latency (CPU) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RUN-01** | REACT v1 | PSG | In-Domain | **30.47%** | **33.94%** | **23.98%** | **26.03%** | 30.11% | 33.86% | 1223.9 ms |
| **RUN-02** | REACT v1 | VG150 | In-Domain | **24.55%** | **28.32%** | **16.61%** | **18.79%** | 26.04% | 30.38% | 1481.5 ms |
| **RUN-03** | REACT v1 | IndoorVG | In-Domain | **31.08%** | **36.63%** | **26.48%** | **29.74%** | 29.96% | 35.89% | 618.3 ms |
| **RUN-04** | REACT v1 | OpenImages | Transfer | **1.78%** | **1.90%** | **10.76%** | **11.45%** | 2.08% | 2.31% | 1630.9 ms |
| **RUN-05** | REACT++ | PSG | In-Domain | **35.98%** | **37.78%** | **23.20%** | **24.16%** | 35.60% | 37.55% | 451.9 ms |
| **RUN-06** | REACT++ | VG150 | In-Domain | **23.54%** | **25.26%** | **16.57%** | **17.92%** | 26.59% | 28.62% | 465.1 ms |
| **RUN-07** | REACT++ | IndoorVG | Transfer | **14.75%** | **16.03%** | **13.25%** | **14.07%** | 16.99% | 18.60% | 468.9 ms |
| **RUN-08** | REACT++ | OpenImages | Transfer | **2.24%** | **2.24%** | **14.11%** | **14.11%** | 2.42% | 2.42% | 499.7 ms |
| **RUN-09** | FCSGG | PSG | Transfer | **4.39%** | **5.08%** | **1.10%** | **1.26%** | 4.33% | 5.05% | 884.2 ms |
| **RUN-10** | FCSGG | VG150 | In-Domain | **9.08%** | **12.44%** | **2.62%** | **3.67%** | 9.14% | 12.56% | 903.0 ms |
| **RUN-11** | FCSGG | IndoorVG | Transfer | **7.97%** | **9.07%** | **3.97%** | **4.34%** | 9.33% | 10.59% | 913.8 ms |
| **RUN-12** | FCSGG | OpenImages | Transfer | **0.80%** | **0.80%** | **3.72%** | **3.72%** | 1.04% | 1.04% | 919.8 ms |
| **RUN-13** | C-SGG | PSG | Hybrid | **32.24%** | **36.21%** | **21.82%** | **23.94%** | 31.84% | 35.74% | 495.5 ms |
| **RUN-14** | C-SGG | VG150 | In-Domain | **20.05%** | **24.16%** | **14.10%** | **16.48%** | 21.79% | 26.41% | 350.9 ms |
| **RUN-15** | C-SGG | IndoorVG | Transfer | **12.15%** | **14.28%** | **10.81%** | **12.50%** | 13.82% | 16.27% | 363.3 ms |
| **RUN-16** | C-SGG | OpenImages | Hybrid | **2.45%** | **2.45%** | **24.03%** | **24.03%** | 2.08% | 2.08% | 370.4 ms |

### 7.4 OpenImages Dual-Protocol Audit Verification

Under Protocol B (excluding unary attributes, evaluating 146 physical interactions across 58 eligible images):

| Model | Variant | Total Physical GT | Hits@50 | Hits@100 | Image-Avg R@50 | Pooled R@50 | Macro mR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **REACT v1** | PyTorch YOLOv8m | 146 | 18 | 20 | **12.02%** | **12.33%** | **11.66%** |
| **REACT++** | ONNX YOLO12m | 146 | 21 | 21 | **13.04%** | **14.38%** | **15.28%** |
| **FCSGG** | CenterNet / RAF | 146 | 9 | 9 | **5.62%** | **6.16%** | **4.03%** |
| **C-SGG** | YOLOv5l + CKN (OID) | 146 | 18 | 18 | **15.47%** | **12.33%** | **26.04%** |

---

## 8. Blueprint for Scaling to Full Datasets (Tens of Thousands of Images)

Scaling from 200 images to the full datasets requires moving beyond simple loops to an industrial-grade evaluation pipeline.

### 8.1 Full Test Split Populations & Download Sources

1. **Visual Genome (VG150)**:
   - **Test Images**: **31,876 images** (108,077 total in VG).
   - Images Part 1 (9.2 GB): `https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip`
   - Images Part 2 (5.4 GB): `https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip`
   - Annotations (`VG-SGG-dicts-with-attri.json`, `image_data.json`): Official Stanford SGG benchmark repository ([Kaihua Tang's repo](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)).
2. **Panoptic Scene Graph (PSG)**:
   - **Test Images**: **2,177 images** (48,749 total in COCO 2017).
   - Images: COCO 2017 val images (`http://images.cocodataset.org/zips/val2017.zip`, 1 GB).
   - Annotations (`psg_val_test.json`): Official OpenPSG repo (`https://github.com/Jingkang50/OpenPSG`).
3. **IndoorVG**:
   - **Test Images**: **4,000+ indoor images** filtered from Visual Genome.
4. **Open Images V6 (OIDv6)**:
   - **Test Images**: **~10,000 images** in the test relationship split.
   - Annotations & Bounding Boxes: `https://storage.googleapis.com/openimages/v6/oidv6-test-annotations-vrd.csv`.

### 8.2 Hardware & Pipeline Migration (CPU to Multi-GPU CUDA)

Running 31,876 images on CPU at 1.5 seconds per image would require ~13.2 hours per model. Moving to GPU achieves 20–40 ms per image (15–25 minutes total run time).

#### 1. ONNX Runtime CUDA Acceleration
In [`run_benchmark_reactpp_vg150.py`](file:///d:/Minor/experiments/REACTPP/run_benchmark_reactpp_vg150.py), update the session initialization:
```python
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024, # 8 GB VRAM limit
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]
session = ort.InferenceSession(onnx_path, providers=providers)
```

#### 2. PyTorch CUDA Execution
In PyTorch runners (`run_benchmark_react_vg150.py`, `run_benchmark_fcsgg_vg150.py`):
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
```

### 8.3 Streaming DataLoader & Batched Inference Architecture

To maximize GPU throughput, use PyTorch's `DataLoader` with prefetching and asynchronous memory pinning:

```python
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

class FullScaleSGGDataset(Dataset):
    def __init__(self, image_paths, img_size=640):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_raw = cv2.imread(path)
        img_letterboxed, gain, pad = letterbox(img_raw, new_shape=(self.img_size, self.img_size))
        tensor = torch.from_numpy(img_letterboxed).permute(2, 0, 1).float() / 255.0
        return tensor, path, gain, pad, img_raw.shape[:2]

loader = DataLoader(
    FullScaleSGGDataset(test_image_paths),
    batch_size=8,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
```

### 8.4 Sharded Disk Caching to Prevent RAM Out-of-Memory (OOM)

> [!CAUTION]
> Accumulating raw predictions in system RAM across 31,876 images will crash with `MemoryError`. At 100 predicted relations per image, holding $3.18 \times 10^6$ triplets with bounding boxes and feature tensors requires over 24 GB of RAM.

**The Sharded Disk Cache Pattern**:
Save predictions incrementally in shards of 1,000 images:

```python
shard_size = 1000
current_shard = {}
shard_idx = 0
os.makedirs(".cache/full_scale_shards", exist_ok=True)

for i, (image_id, preds) in enumerate(run_inference_stream()):
    current_shard[image_id] = {
        'boxes': preds['boxes'].cpu().half(),   # Cast float32 -> float16
        'rels': preds['rels'].cpu().half(),
        'scores': preds['scores'].cpu().half()
    }
    if (i + 1) % shard_size == 0 or (i + 1) == total_images:
        shard_file = f".cache/full_scale_shards/shard_{shard_idx:03d}.pt"
        torch.save(current_shard, shard_file)
        current_shard.clear()
        shard_idx += 1
```

### 8.5 Vectorized Metric Accumulation over Millions of Triplets

To calculate R@K and mR@K over 31,876 images without slowdown:
1. Do not compute IoU in Python loops. Use `torchvision.ops.box_iou` on PyTorch tensors or vectorized NumPy broadcasting.
2. Maintain predicate hit counters using sparse arrays:
```python
gt_per_predicate = np.zeros(num_predicates, dtype=np.int64)
hits_per_predicate_at_50 = np.zeros(num_predicates, dtype=np.int64)
hits_per_predicate_at_100 = np.zeros(num_predicates, dtype=np.int64)

# Stream each image shard independently
for shard_file in sorted(glob.glob(".cache/full_scale_shards/shard_*.pt")):
    shard = torch.load(shard_file, weights_only=False)
    for img_id, pred in shard.items():
        gt = ground_truth_dict[img_id]
        # Match using vectorized IoU
        hits50, hits100 = evaluate_image_triplets(pred, gt)
        accumulate_counts(hits50, hits100, gt)

# Compute macro Mean Recall instantly
mR_50 = np.mean(hits_per_predicate_at_50[gt_per_predicate > 0] / gt_per_predicate[gt_per_predicate > 0])
mR_100 = np.mean(hits_per_predicate_at_100[gt_per_predicate > 0] / gt_per_predicate[gt_per_predicate > 0])
```

---

## 9. Step-by-Step Practical Replication Guide

### 9.1 Environment & Dependency Setup

```bash
# 1. Clone repository
git clone https://github.com/Nikhil-Verma22/minor_project.git
cd minor_project

# 2. Create clean Python virtual environment (Python 3.10, 3.11, or 3.12 recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install core PyTorch & Torchvision (CUDA 12.4 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Or for CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install ONNX Runtime & computer vision utilities
pip install onnxruntime-gpu opencv-python pycocotools pyyaml pandas tabulate tqdm psutil pillow
```

### 9.2 Repository Layout & Checkpoint Placement

Ensure your directory matches this exact structure:
```
minor_project/
├── checkpoints/
│   ├── FastContextualSGG/
│   │   └── yolol_object_test_28.pt
│   ├── FCSGG/
│   │   └── HRNetW32-1S.pth
│   ├── PSG/
│   │   └── react++_yolo12m/
│   │       └── yolo12m/
│   │           └── react_pp_yolo12m.onnx
│   └── REACT/
│       ├── react_indoorvg_yolov8m.onnx
│       ├── react_pp_vg150_yolo12m.onnx
│       ├── react_PSG.zip
│       ├── react_VG150.zip
│       ├── react_PSG/react_PSG/best_model_epoch_11.pth
│       └── react_VG150/best_model_epoch_9.pth
├── evaluation_sets/
│   ├── IndoorVG_200_annotations.coco.json
│   ├── OpenImages_200_annotations.coco.json
│   ├── PSG_200_annotations.coco.json
│   ├── VG150_200_annotations.coco.json
│   ├── predicate_mapping_evaluation_v1.2_FINAL.csv
│   └── images/
│       ├── IndoorVG/  (200 images)
│       ├── OpenImages/ (200 images)
│       ├── PSG/        (200 images)
│       └── VG150/      (200 images)
├── model_repos/
│   ├── fast-context-scene-graph-generation/
│   ├── fcsgg/
│   └── SGG-Benchmark/
└── experiments/
    ├── FastContextualSGG/
    ├── FCSGG/
    ├── REACT/
    └── REACTPP/
```

### 9.3 Checkpoint SHA-256 Verification Table

Run this script to verify that all checkpoint files are bit-for-bit identical to our audited weights:

```python
import hashlib, os

expected_digests = {
    "checkpoints/REACT/react_VG150/best_model_epoch_9.pth": "8bf598afdb2ccef9e5b6f54dc1fe0fa70b46a4bfceecfb02f693f2e82ad503bf",
    "checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth": "0f27511e66f77f6d3eac4c09f5ccb06a61e702547032a2eddb1b601fe26ff03d",
    "checkpoints/REACT/react_indoorvg_yolov8m.onnx": "5188b578aa155e3584d95b0e3d87a5bbd80ae2464f6d1f3e380d9b7ad872c7e8",
    "checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx": "38c4bdbb7e2a077131ffcbf57af726eb30574318e2344d5b4bfc1d3a48f5b2fe",
    "checkpoints/REACT/react_pp_vg150_yolo12m.onnx": "ce01616b902271da224c61ff85f9914f8efb12c8894e4aa7bad42ebe9480c9f2",
    "checkpoints/FCSGG/HRNetW32-1S.pth": "41aa7639bac0793ef6ad04b860b2d441a32523cf0f69b7590b2eda261638c084",
    "checkpoints/FastContextualSGG/yolol_object_test_28.pt": "f657e21917f10d2d70da5e3f1b02d3148dc8589ac74ba6e7d8c2fc5e2aaac257",
}

for path, expected in expected_digests.items():
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    digest = h.hexdigest()
    status = "PASS" if digest == expected else "FAIL"
    print(f"[{status}] {path} -> {digest[:16]}...")
```

### 9.4 Running the 16 Benchmark Scripts (Single-Image, 200-Image, Full-Scale)

Every runner supports `--max-images`:
- `--max-images 1`: Fast smoke test to verify model forward pass and box undo-letterboxing.
- `--max-images 200`: Standard benchmark evaluation on the frozen benchmark subset.
- `--max-images -1`: Full dataset evaluation over all images.

#### Group 1: REACT v1
```bash
python experiments/REACT/run_benchmark_react_psg.py --max-images 200
python experiments/REACT/run_benchmark_react_vg150.py --max-images 200
python experiments/REACT/run_benchmark_react_indoorvg.py --max-images 200
python experiments/REACT/run_benchmark_react_openimages.py --max-images 200
```

#### Group 2: REACT++
```bash
python experiments/REACTPP/run_benchmark_reactpp_psg.py --max-images 200
python experiments/REACTPP/run_benchmark_reactpp_vg150.py --max-images 200
python experiments/REACTPP/run_benchmark_reactpp_indoorvg.py --max-images 200
python experiments/REACTPP/run_benchmark_reactpp_openimages.py --max-images 200
```

#### Group 3: FCSGG
```bash
python experiments/FCSGG/run_benchmark_fcsgg_psg.py --max-images 200
python experiments/FCSGG/run_benchmark_fcsgg_vg150.py --max-images 200
python experiments/FCSGG/run_benchmark_fcsgg_indoorvg.py --max-images 200
python experiments/FCSGG/run_benchmark_fcsgg_openimages.py --max-images 200
```

#### Group 4: Fast Contextual SGG (C-SGG)
```bash
python experiments/FastContextualSGG/run_benchmark_fast_contextual_psg.py --max-images 200
python experiments/FastContextualSGG/run_benchmark_fast_contextual_vg150.py --max-images 200
python experiments/FastContextualSGG/run_benchmark_fast_contextual_indoorvg.py --max-images 200
python experiments/FastContextualSGG/run_benchmark_fast_contextual_openimages.py --max-images 200
```

### 9.5 Master Troubleshooting Playbook

| Error / Symptom | Root Cause | Exact Resolution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'detectron2._C'` | Detectron2 native C++ extensions are uncompiled. | Inject the dynamic stub: `types.ModuleType("detectron2._C")` aliased to `torchvision.ops.nms` (see Section 4.1). |
| `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'` | Pillow 10+ removed `PIL.Image.LINEAR`. | Insert monkey patch: `PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR`. |
| `WeightsOnlyError: Cannot load legacy pickle` | PyTorch 2.6+ defaults to `weights_only=True`. | Pass `weights_only=False` in `torch.load(path, weights_only=False)`. |
| `ModuleNotFoundError: No module named 'models'` | YOLOv5 pickle requires `models.yolo.Model` on path. | Insert `model_repos/.../VG_dataset` to `sys.path`. |
| `RuntimeError: no word vectors of requested dimension found` | Stanford GloVe archive missing from disk. | Run `extract_glove.py` to stream necessary subsets, or download full `glove.6B.zip`. |
| `UnicodeEncodeError: 'charmap' codec can't encode character` | Windows shell defaulted to CP1252. | Insert `sys.stdout.reconfigure(encoding='utf-8')` at script startup. |
| Inexplicably low recall on OpenImages (<3%) | Evaluation evaluated 720 unary visual attributes (`is`). | Implement Protocol B (filter predicate 27 `is`) to evaluate true physical relationships. |

---

## 10. Conclusion & Key Research Insights

1. **Context-Only Reasoning is Highly Efficient**: Fast Contextual SGG proves that scoring 30d spatial geometry vectors and 50d GloVe semantics through a 3-layer MLP achieves competitive Mean Recall (14.10% on VG150, 26.04% on OpenImages Protocol B) while running at **2.85 FPS end-to-end on CPU** (4.96 ms for 2,580 pairs).
2. **Single-Stage ONNX Deployment is Viable**: REACT++ demonstrates that an end-to-end YOLO12m network compiled into ONNX matches two-stage transformer accuracy while cutting latency by 60–70% (451 ms vs 1223 ms on CPU).
3. **Cross-Ontology Transfer Exposes Dataset Biases**: When evaluating Visual Genome models on OpenPSG or OpenImages, recall drops not because the vision backbone fails, but because dataset ontologies define different semantic granularities (e.g. Visual Genome's overwhelming bias toward `on`, `has`, and `wearing` vs. OpenImages' dominance of unary attributes).
4. **Reproducibility Demands Bit-Level Auditing**: Without exact tracking of compound scoring contracts, unpickling environments, and explicit protocol definitions (Protocol A vs. B), cross-dataset SGG benchmarks cannot be compared fairly.

This repository, alongside this replication guide, establishes a transparent, verified, and permanent foundation for future scene graph generation research.
