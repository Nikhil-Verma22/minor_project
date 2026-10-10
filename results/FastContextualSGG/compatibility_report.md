# Fast Contextual Scene Graph Generation (C-SGG & CV-SGG): Inspection and Compatibility Report

**Workspace**: `d:/Minor`  
**Repository**: `model_repos/fast-context-scene-graph-generation/`  
**Upstream**: [https://github.com/moshuilanting/fast-context-scene-graph-generation](https://github.com/moshuilanting/fast-context-scene-graph-generation)  
**Associated Paper**: *"Fast Contextual Scene Graph Generation with Unbiased Context Augmentation"* (Jin et al.)  
**Benchmark Target**: Object Retrieval in Dynamic Workspace Environments  
**Date**: October 10, 2026  

---

## 1. Repository Identity & Paper Verification

- **Identity Confirmation**:
  - The repository `moshuilanting/fast-context-scene-graph-generation` is the official implementation of *"Fast Contextual Scene Graph Generation with Unbiased Context Augmentation"*.
  - It defines the context-first paradigm where relationship candidates are initially predicted using object semantic categories and spatial context representations, followed by visual verification.
  - Two distinct operational variants are defined by the authors:
    1. **C-SGG (Contextual-only Scene Graph Generation)**: Predicts relationships solely from semantic word embeddings and bounding box spatial configurations via the **Context Knowledge Network (CKN)**.
    2. **CV-SGG (Contextual + Visual Scene Graph Generation)**: Uses CKN as an initial contextual filter to prune unpromising relationship candidates, followed by the **Vision Differentiation Network (VDN)** to visually verify candidate relationships against image pixel crops with subject/object binary masks.

---

## 2. Architecture & Pipeline Breakdown

### 2.1 Context Knowledge Network (CKN / C-SGG)
- **Input Representation (190 dimensions)**:
  - Subject and object category semantic embeddings: 50-dimensional GloVe vectors ($2 \times 50 = 100$ dims).
  - Subject and object spatial configuration features: 6-dimensional normalized spatial vector $[\Delta x/2, \Delta y/2, x_1, y_1, x_2, y_2]$ repeated 5 times ($2 \times 30 = 60$ dims).
  - Spatial difference vector: $\text{loc}_{\text{obj}} - \text{loc}_{\text{sub}}$ ($30$ dims).
  - Total context vector size: $100 + 60 + 30 = 190$ dims.
- **Model Architecture**:
  - 3-layer MLP: $\text{Linear}(190 \to 256) \to \text{ReLU} \to \text{Linear}(256 \to 512) \to \text{ReLU} \to \text{Linear}(512 \to 51)$.
  - Total parameters: $331,827$ (~1.3 MB).
  - Extremely fast CPU/GPU inference ($< 1$ ms per image for all pairs).

### 2.2 Vision Differentiation Network (VDN / CV-SGG)
- **Two-Stage Pipeline**:
  - **Stage 1 (Contextual Filtering)**: CKN evaluates all $N(N-1)$ directed object pairs. Only the top $K$ candidate pairs (default $K=150$ or $300$) with the highest contextual relationship scores are retained.
  - **Stage 2 (Visual Verification)**: For each retained candidate pair, an image patch is cropped to the union bounding box of the subject and object, resized to $224 \times 224$, and combined with subject and object binary indicator masks into a 5-channel tensor $[M_{\text{sub}}, R, G, B, M_{\text{obj}}]$.
  - **Visual Feature Extraction & Fusion**:
    - Custom ResNet backbone with Bottleneck layers $[2, 2, 2, 2, 2]$ extracts visual features.
    - Two-layer semantic MLP projects word vectors into 256 dimensions.
    - Fusion layer combines visual tokens with semantic embeddings to yield visual relation logits:
      $$\text{Score}_{\text{final}} = 0.5 \cdot \sigma(\text{Logits}_{\text{visual}}) + 0.5 \cdot \sigma(\text{Logits}_{\text{context}})$$
- **Total parameters**: $4,785,967$ (~19.2 MB).

---

## 3. Pretrained Checkpoint & Dependency Inventory

| Component / File | Checkpoint Path | Status | Size | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **VG CKN (C-SGG)** | `VG_dataset/ckn/CKN+EPBS.pt` | **PRESENT** | 828 KB | Verified loading in PyTorch CPU. Clean state dict. |
| **VG VDN (CV-SGG)** | `VG_dataset/vdn/CKN+VDN+EPBS.pt` | **PRESENT** | 19.2 MB | Verified loading in PyTorch CPU. Full weights present. |
| **VG Object Detector** | `yolol_object_test_28.pt` | **REMOTE (GDrive)** | 359 MB (376 MB) | Official YOLOv5l trained on VG150. Verified downloadable from official link. |
| **GloVe 50d Vectors** | `.cache/FastContextualSGG/vg_word_vec_list.pt` | **CACHED** | 30 KB (151 tokens) | Extracted and cached locally without deprecated torchtext. |
| **PSG CKN** | `PSG_dataset/ckn/2619.pt` | **PRESENT** | 840 KB | Loads on PyTorch CPU. |
| **PSG VDN** | `PSG_dataset/vdn/base_last.pt` | **PRESENT** | 19.2 MB | Loads on PyTorch CPU. |
| **PSG Panoptic Masks** | `SegFromer_PVTV/` | **MISSING OFFLINE** | N/A | Requires offline precomputed SegFormer PVTv2 masks. |
| **OIDv6 CKN** | `OID_dataset/ckn/ckn_oid_EPBS_n.pt` | **PRESENT** | 787 KB | Loads on PyTorch CPU. |
| **OIDv6 VDN** | `OID_dataset/vdn/EPBS.../last.pt` | **PRESENT** | 39.4 MB | Loads on PyTorch CPU. |
| **OIDv6 Detection Boxes** | `openimage_v6_test/bbox_save.bin` | **MISSING OFFLINE** | N/A | Requires offline precomputed PySGG Faster R-CNN boxlists. |

---

## 4. Dataset Compatibility Matrix

| Dataset | Split | C-SGG Status | CV-SGG Status | Reason / Compatibility Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Visual Genome (VG150)** | 200 frozen imgs | **SUPPORTED** | **SUPPORTED** | Native 150 object classes, 50 predicate classes. Full CKN & VDN weights available. Detector `yolol_object_test_28.pt` (359MB) is available. |
| **OpenPSG (PSG)** | 200 frozen imgs | **BLOCKED** | **BLOCKED** | Implementation does NOT contain an online detector; strictly depends on precomputed offline SegFormer PVTv2 panoptic segmentation files (`SegFromer_PVTV/`). |
| **Open Images V6 (OIDv6)**| 200 frozen imgs | **BLOCKED** | **BLOCKED** | Implementation strictly depends on precomputed offline Faster R-CNN pickle files (`bbox_save.bin`). No online detector provided in repository. |
| **IndoorVG** | 200 frozen imgs | **BLOCKED (LEAKAGE)**| **BLOCKED (LEAKAGE)**| 100% contaminated with VG150 training split (`LEAK_INDOOR_VGTRAIN`). No method-specific checkpoints or adaptations exist. |

---

## 5. Execution Strategy

1. **Download YOLOv5l Detector**:
   - Download official `yolol_object_test_28.pt` (359 MB) into `checkpoints/FastContextualSGG/yolol_object_test_28.pt` (or link to `model_repos/fast-context-scene-graph-generation/VG_dataset/`).
2. **Implement Unified Adapter**:
   - `experiments/FastContextualSGG/fast_contextual_adapter.py`:
     - Encapsulates YOLOv5l object detection (`conf_thres=0.01`, `iou_thres=0.45`, max 80 objects).
     - **Variant 1 (C-SGG)**: Runs CKN on all object pairs; outputs context relation predictions and scores.
     - **Variant 2 (CV-SGG)**: Runs CKN contextual candidate filtering (top 150/300 pairs); runs VDN visual verification on union crops; outputs fused relationship predictions.
     - Measures detailed timing breakdown: Contextual Latency, Visual Verification Latency, End-to-End Latency, Candidate Pair Counts.
3. **Unit Tests & Regression Verification**:
   - Test CKN prediction logic, VDN forward pass, coordinate transforms, native predicate mapping, and ontology projection across strict, extended, tri-dataset, and native views.
4. **Smoke Test (3 Images)**:
   - Run on 3 sample VG150 images, inspect output scene graphs, and confirm stability before the full benchmark.
5. **Full 200-Image Benchmark Execution**:
   - Run both C-SGG and CV-SGG on the 200-image frozen Visual Genome evaluation set across all 4 ontology views.
6. **Data Integrity & Final Deliverables**:
   - Verify SHA-256 integrity against Prompt 1 baseline (`results/REACT/eval_sets_sha256_post.json`).
   - Generate all artifacts, CSV/JSON metrics, visualizations, and comprehensive benchmark report.
