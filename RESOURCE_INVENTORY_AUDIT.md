# Academic Benchmark Resource Inventory & Checkpoint Audit

**Project**: *Cross-Dataset Scene Graph Generation (SGG) Benchmark Suite*  
**Models Audited**: REACT v1, REACT++, FCSGG, and Fast Contextual SGG (C-SGG)  
**Target Datasets**: Panoptic Scene Graph (PSG), Visual Genome (VG150), IndoorVG, and Open Images V6 (OIDv6)  
**Host Machine**: Intel(R) Core(TM) Ultra 5 125H (14 cores, 18 threads, 16 GB RAM), Windows 11 Home 64-bit  
**Primary Repository**: [`d:\Minor`](file:///d:/Minor) | **Secondary Repository**: [`d:\Minor_2`](file:///d:/Minor_2)  
**Audit Protocol**: Strict read-only physical disk audit. Zero network downloads triggered; zero files moved, renamed, or modified.

---

## Table of Contents

1. [Executive Summary & Verification Status](#1-executive-summary--verification-status)
2. [GloVe Word Embeddings Inventory (Priority 1)](#2-glove-word-embeddings-inventory-priority-1)
   - [2.1 Search Results Across Common Locations](#21-search-results-across-common-locations)
   - [2.2 Technical Analysis of REACT v1 (200d) Requirement](#22-technical-analysis-of-react-v1-200d-requirement)
   - [2.3 Technical Analysis of FCSGG / C-SGG (50d) Requirement](#23-technical-analysis-of-fcsgg--c-sgg-50d-requirement)
3. [REACT v1 Checkpoints](#3-react-v1-checkpoints)
   - [3.1 Model Architecture & Distinction from REACT++](#31-model-architecture--distinction-from-react)
   - [3.2 Visual Genome (VG150) Checkpoint](#32-visual-genome-vg150-checkpoint)
   - [3.3 Panoptic Scene Graph (PSG) Checkpoint](#33-panoptic-scene-graph-psg-checkpoint)
   - [3.4 IndoorVG Model & Missing PyTorch Weights](#34-indoorvg-model--missing-pytorch-weights)
4. [FCSGG Resources](#4-fcsgg-resources)
   - [4.1 Architecture & Detector Clarification (CenterNet vs. YOLO)](#41-architecture--detector-clarification-centernet-vs-yolo)
   - [4.2 Checkpoints, Configurations & Upstream Provenance](#42-checkpoints-configurations--upstream-provenance)
   - [4.3 CPU Compatibility Adapter & Execution Stub](#43-cpu-compatibility-adapter--execution-stub)
5. [REACT++ Resources](#5-react-resources)
   - [5.1 Model Architecture (YOLO12m Single-Stage ONNX)](#51-model-architecture-yolo12m-single-stage-onnx)
   - [5.2 Dataset Exports & Transfer Strategies](#52-dataset-exports--transfer-strategies)
6. [Fast Contextual SGG (C-SGG) Resources](#6-fast-contextual-sgg-c-sgg-resources)
7. [Download History, Shell Logs & Source Provenance](#7-download-history-shell-logs--source-provenance)
8. [Master Resource Audit Matrix](#8-master-resource-audit-matrix)
   - [Category 1: Files Physically Present on Disk](#category-1-files-physically-present-on-disk)
   - [Category 2: Files Available Only in Caches / Subsets](#category-2-files-available-only-in-caches--subsets)
   - [Category 3: Files Downloadable from Verified Upstream Sources](#category-3-files-downloadable-from-verified-upstream-sources)
   - [Category 4: Genuinely Missing / Non-Existent Files](#category-4-genuinely-missing--non-existent-files)
9. [Handoff, Packaging & Safe Transfer Guide](#9-handoff-packaging--safe-transfer-guide)

---

## 1. Executive Summary & Verification Status

This audit comprehensively documents every pretrained weight file, configuration, word vector cache, and supporting script required to execute the 16-run Scene Graph Generation benchmark across **REACT v1**, **REACT++**, **FCSGG**, and **Fast Contextual SGG (C-SGG)**.

### Headline Findings:
1. **GloVe Embeddings**:
   - The full Stanford NLP archive (`glove.6B.zip`, 862 MB) and original text files (`glove.6B.*d.txt`) are **not physically present on disk**.
   - The file at [`d:\Minor\.cache\REACT\glove\glove.6B.200d.pt`](file:///d:/Minor/.cache/REACT/glove/glove.6B.200d.pt) is a **2,387-byte stub** created to bypass the online download check in `txt_embeddings.py`. It contains an empty token mapping (`{}`), causing REACT v1 to initialize token embeddings via random Gaussian initialization (`normal_(0, 1)`).
   - Fast Contextual SGG solved its 50d GloVe requirement by streaming only the needed vocabulary (215 words) from Hugging Face into compact tensor files: [`glove_extracted_50d.pt`](file:///d:/Minor/.cache/FastContextualSGG/glove_extracted_50d.pt), [`vg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/vg_word_vec_list.pt), and [`psg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/psg_word_vec_list.pt).
2. **REACT v1 Checkpoints**:
   - Authentic official upstream checkpoints for **VG150** (Epoch 9, 225.05 MB) and **PSG** (Epoch 11, 225.00 MB) are present and verified loadable in PyTorch.
   - For **IndoorVG**, the native model exists as an ONNX graph (`react_indoorvg_yolov8m.onnx`, 132.45 MB); the raw PyTorch `.pth` checkpoint is missing locally but downloadable from Hugging Face.
3. **FCSGG Resources**:
   - The official CVPR 2021 checkpoint [`HRNetW32-1S.pth`](file:///d:/Minor/checkpoints/FCSGG/HRNetW32-1S.pth) (362.41 MB) is present and verified.
   - **Crucial Architectural Note**: FCSGG is an anchor-free bottom-up network using **CenterNet keypoints** and **Relation Affinity Fields (RAF)** directly on an **HRNet-W32** backbone. It **does not use YOLO detector weights**.
4. **REACT++ Resources**:
   - Single-stage YOLO12m ONNX models for **PSG** (111.38 MB) and **VG150** (111.57 MB) are present, verified, and loadable via ONNX Runtime.

---

## 2. GloVe Word Embeddings Inventory (Priority 1)

### 2.1 Search Results Across Common Locations

An exhaustive search was conducted across project repositories, user download directories, desktop, user cache folders, Conda/Python environments, and browser download logs.

| Query Target | Found on Disk? | Exact Local File Path | File Size | Classification | Complete & Readable? | SHA-256 Digest |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `glove.6B.zip` | **NO** | *None* | — | Archive | No (Missing) | — |
| `glove.6B.200d.txt` | **NO** | *None* | — | Raw Text | No (Missing) | — |
| `glove.6B.50d.txt` | **NO** | *None* | — | Raw Text | No (Missing) | — |
| `glove.6B.100d.txt` | **NO** | *None* | — | Raw Text | No (Missing) | — |
| `glove.6B.300d.txt` | **NO** | *None* | — | Raw Text | No (Missing) | — |
| `glove.6B.200d.pt` | **YES** | [`d:\Minor\.cache\REACT\glove\glove.6B.200d.pt`](file:///d:/Minor/.cache/REACT/glove/glove.6B.200d.pt) | 2,387 bytes | PyTorch Cache | **NO (Empty Stub)**<br>Readable tuple `({}, zeros(1,200), 200)` | `0c401134071e39c1b95de7420863a340dcf38c8fb0f53b4e205ad0694c4829c3` |
| `glove.6B.50d.pt` | **NO** | *None* | — | PyTorch Cache | No (Full cache missing) | — |
| `glove_extracted_50d.pt` | **YES** | [`d:\Minor\.cache\FastContextualSGG\glove_extracted_50d.pt`](file:///d:/Minor/.cache/FastContextualSGG/glove_extracted_50d.pt) | 103,917 bytes | Subset Cache | **YES (Target Words Only)**<br>215 words mapped to 50d tensors | `e44525ce7f09cad236498b4e097ec48d9ba146e747b7cccef59f6e75fe760da3` |
| `vg_word_vec_list.pt` | **YES** | [`d:\Minor\.cache\FastContextualSGG\vg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/vg_word_vec_list.pt) | 31,848 bytes | Tensor Cache | **YES**<br>Tensor shape `(151, 50)` float32 | `a3bc639414c53b20d2b48a1eff8efb55a46ab05adbc59137f625c8d085ccd47d` |
| `psg_word_vec_list.pt` | **YES** | [`d:\Minor\.cache\FastContextualSGG\psg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/psg_word_vec_list.pt) | 28,399 bytes | Tensor Cache | **YES**<br>Tensor shape `(134, 50)` float32 | `dd42ac400e8e6c41f3cfbac235848f03f456c58494a26ed2fe5824bffc44be22` |

### 2.2 Technical Analysis of REACT v1 (200d) Requirement

In upstream SGG-Benchmark / REACT v1, [`sgg_benchmark/utils/txt_embeddings.py`](file:///d:/Minor/model_repos/SGG-Benchmark/sgg_benchmark/utils/txt_embeddings.py#L251-L316) defines the word vector loader:
```python
def load_word_vectors(root, wv_type, dim):
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        return torch.load(fname + '.pt', map_location=torch.device("cpu"))
    # If missing, falls back to downloading glove.6B.zip (862 MB) from Stanford NLP
```
When `obj_edge_vectors` and `rel_vectors` are called in [`react_predictor.py`](file:///d:/Minor/model_repos/SGG-Benchmark/sgg_benchmark/modeling/roi_heads/relation_head/predictors/react_predictor.py#L67-L68):
```python
vectors = torch.Tensor(len(names), wv_dim)  # 51, 200
vectors.normal_(0, 1)
for i, token in enumerate(names):
    wv_index = wv_dict.get(token, None)
    if wv_index is not None:
        vectors[i] = wv_arr[wv_index]
    # else: keeps the normal initialization
```
**Critical Finding**:  
Our disk inspection of [`d:\Minor\.cache\REACT\glove\glove.6B.200d.pt`](file:///d:/Minor/.cache/REACT/glove/glove.6B.200d.pt) reveals:
- Structure: Tuple of `(wv_dict, wv_arr, wv_size)`
- `type(wv_dict)`: `dict`, **length: 0** (`{}`)
- `type(wv_arr)`: `torch.Tensor`, **shape: [1, 200]** (zeros)
- `wv_size`: `200`
- **Effect**: Because `wv_dict` is empty, `wv_dict.get(token)` consistently returns `None`. Consequently, REACT v1 retains its random Gaussian normal initialization `normal_(0, 1)` for all entity class embeddings. It bypassed network downloads, but **does not contain true GloVe word vectors**.

### 2.3 Technical Analysis of FCSGG / C-SGG (50d) Requirement

1. **FCSGG**:
   - The upstream FCSGG inference engine (`fcsgg/modeling/roi_heads/relation_head/relation_head.py`) uses visual heatmaps and RAF feature fields. It does not query GloVe text embeddings during forward prediction.
2. **Fast Contextual SGG (C-SGG)**:
   - C-SGG (*Jin et al., CVPR 2023*) explicitly requires 50d GloVe word vectors for subject and object classes.
   - Script [`d:\Minor\.cache\FastContextualSGG\extract_glove.py`](file:///d:/Minor/.cache/FastContextualSGG/extract_glove.py) streams `https://huggingface.co/JeremiahZ/glove/resolve/main/glove.6B.50d.txt`, selectively filtering only the 215 unique category names needed for Visual Genome and OpenPSG.
   - It saved:
     - [`glove_extracted_50d.pt`](file:///d:/Minor/.cache/FastContextualSGG/glove_extracted_50d.pt) (215 words $\times$ 50d tensors)
     - [`vg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/vg_word_vec_list.pt) (151 classes $\times$ 50d tensor, SHA-256: `a3bc6394...`)
     - [`psg_word_vec_list.pt`](file:///d:/Minor/.cache/FastContextualSGG/psg_word_vec_list.pt) (134 classes $\times$ 50d tensor, SHA-256: `dd42ac40...`)
   - **Conclusion**: C-SGG has fully functional, complete, and verified 50d GloVe semantics for all supported benchmark datasets.

---

## 3. REACT v1 Checkpoints

### 3.1 Model Architecture & Distinction from REACT++

> [!IMPORTANT]
> **Definitive Architectural Separation**:
> - **REACT v1**: Two-stage modular framework. Stage 1 extracts bounding boxes and visual features via a **PyTorch YOLOv8m** detector. Stage 2 executes relational reasoning through an explicit **Relation Transformer Head** (RoI feature projection, GloVe word embeddings, multi-head contextual attention).
> - **REACT++**: Single-stage fully integrated framework. Built upon **YOLO12m** with Detection-Anchored Multi-Scale Pooling (DAMP) and an AIFI scene context branch, compiled directly into an end-to-end **ONNX** compute graph.

### 3.2 Visual Genome (VG150) Checkpoint

- **Archive File**: [`d:\Minor\checkpoints\REACT\react_VG150.zip`](file:///d:/Minor/checkpoints/REACT/react_VG150.zip) (183,698,786 bytes, 175.19 MB)
  - SHA-256: `e6c9082bdaf1614b6e3345df00236d6177c6212290023456dd005397c4254b4e`
- **Extracted Model Checkpoint**: [`d:\Minor\checkpoints\REACT\react_VG150\best_model_epoch_9.pth`](file:///d:/Minor/checkpoints/REACT/react_VG150/best_model_epoch_9.pth) (235,981,109 bytes, 225.05 MB)
  - SHA-256: `8bf598afdb2ccef9e5b6f54dc1fe0fa70b46a4bfceecfb02f693f2e82ad503bf`
  - Load Test: **PASS** (`torch.load` successful on CPU). State dict keys: `['model', 'optimizer', 'iteration']` with 546 subkeys in `model`.
  - Architecture: PyTorch YOLOv8m + Relation Transformer Head.
  - Training Epoch: **Epoch 9** (Completed 2024-11-14 14:18 by Maelic Neau on a single cluster GPU).
  - Test Metrics from Log: R@20: 21.04%, R@50: 26.16%, R@100: 28.75%, mR@50: 12.26%, mR@100: 13.63%.
- **Accompanying Artifacts Present in Directory**:
  - [`config.yml`](file:///d:/Minor/checkpoints/REACT/react_VG150/config.yml) (6,615 bytes, SHA-256: `8a183592...`)
  - [`labels.json`](file:///d:/Minor/checkpoints/REACT/react_VG150/labels.json) (2,604 bytes, SHA-256: `4a9b646c...`)
  - [`log.txt`](file:///d:/Minor/checkpoints/REACT/react_VG150/log.txt) (123,716 bytes, SHA-256: `4a1d48c0...`)
  - [`VG150_statistics.cache`](file:///d:/Minor/checkpoints/REACT/react_VG150/VG150_statistics.cache) (13.96 MB, SHA-256: `48b1d9bf...`)

### 3.3 Panoptic Scene Graph (PSG) Checkpoint

- **Archive File**: [`d:\Minor\checkpoints\REACT\react_PSG.zip`](file:///d:/Minor/checkpoints/REACT/react_PSG.zip) (200,749,745 bytes, 191.45 MB)
  - SHA-256: `3c0f92adc12bd38c95da358052a6f1d20fd5c3d09d4b9da79d578e3b5e628920`
- **Extracted Model Checkpoint**: [`d:\Minor\checkpoints\REACT\react_PSG\react_PSG\best_model_epoch_11.pth`](file:///d:/Minor/checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth) (235,926,680 bytes, 225.00 MB)
  - SHA-256: `0f27511e66f77f6d3eac4c09f5ccb06a61e702547032a2eddb1b601fe26ff03d`
  - Load Test: **PASS** (`torch.load` successful on CPU). State dict keys: `['model', 'optimizer', 'iteration']` with 546 subkeys in `model`.
  - Architecture: PyTorch YOLOv8m + Relation Transformer Head.
  - Training Epoch: **Epoch 11** (Completed 2024-12-09 20:02 by Maelic Neau).
  - Test Metrics from Log: R@20: 28.46%, R@50: 31.88%, R@100: 33.37%, mR@50: 19.81%, mR@100: 21.89%.
- **Accompanying Artifacts Present in Directory**:
  - [`config.yml`](file:///d:/Minor/checkpoints/REACT/react_PSG/react_PSG/config.yml) (6,635 bytes, SHA-256: `bb16b6ad...`)
  - [`labels.json`](file:///d:/Minor/checkpoints/REACT/react_PSG/react_PSG/labels.json) (2,509 bytes, SHA-256: `ceb4ff56...`)
  - [`log.txt`](file:///d:/Minor/checkpoints/REACT/react_PSG/react_PSG/log.txt) (56,952 bytes, SHA-256: `e06da8d3...`)
  - [`PSG_train_statistics.cache`](file:///d:/Minor/checkpoints/REACT/react_PSG/react_PSG/PSG_train_statistics.cache) (78.88 MB, SHA-256: `35ff7e59...`)

### 3.4 IndoorVG Model & Missing PyTorch Weights

- **Standalone ONNX Model Present**: [`d:\Minor\checkpoints\REACT\react_indoorvg_yolov8m.onnx`](file:///d:/Minor/checkpoints/REACT/react_indoorvg_yolov8m.onnx)
  - Size: 138,878,957 bytes (132.45 MB)
  - SHA-256: `5188b578aa155e3584d95b0e3d87a5bbd80ae2464f6d1f3e380d9b7ad872c7e8`
  - Load Test: **PASS** (`onnxruntime.InferenceSession` with `CPUExecutionProvider`).
  - Embedded Metadata: 84 IndoorVG object classes and 37 predicates.
  - Backbone: `yolov8m` (confirming it is the REACT v1 model family).
  - Source URL: `https://huggingface.co/maelic/REACTPlusPlus_IndoorVG/resolve/main/yolov8m/model.onnx`.
- **PyTorch `.pth` Checkpoint**:
  - Status: **MISSING on disk**. Can be downloaded from `https://huggingface.co/maelic/REACTPlusPlus_IndoorVG/resolve/main/yolov8m/best_model.pth`.

---

## 4. FCSGG Resources

### 4.1 Architecture & Detector Clarification (CenterNet vs. YOLO)

> [!NOTE]
> **Detector Clarification**:
> FCSGG (*Fully Convolutional Scene Graph Generation*, Liu et al., CVPR 2021) is a single-stage anchor-free bottom-up network. It directly detects objects via **CenterNet keypoint estimation** (center heatmaps, bounding box size regression, offset heads) and predicts relationships using **Relation Affinity Fields (RAF)** over an **HRNet-W32** backbone.
> **FCSGG does not utilize YOLO detector weights.** It has its own integrated detector head inside HRNet.

### 4.2 Checkpoints, Configurations & Upstream Provenance

- **Pretrained Checkpoint**: [`d:\Minor\checkpoints\FCSGG\HRNetW32-1S.pth`](file:///d:/Minor/checkpoints/FCSGG/HRNetW32-1S.pth)
  - Size: 380,012,898 bytes (362.41 MB)
  - SHA-256: `41aa7639bac0793ef6ad04b860b2d441a32523cf0f69b7590b2eda261638c084`
  - Load Test: **PASS** (`torch.load` successful on CPU). State dict keys: `['model', 'optimizer', 'scheduler', 'iteration']` with 1,936 subkeys in `model`, iteration: 54,999.
  - Provenance: Downloaded from the official Google Drive provided in the authors' CVPR 2021 repository: `https://drive.google.com/file/d/1IpSDAZQW8irJyfWW5jep9fBVPRZcsTzl/view`.
  - Target Dataset: Visual Genome (VG150: 150 object classes, 50 predicate classes).
- **Official YAML Configuration**: [`model_repos/fcsgg/configs/FCSGG_HRNet_W32_2xDownRAF_512x512_MS.yaml`](file:///d:/Minor/model_repos/fcsgg/configs/FCSGG_HRNet_W32_2xDownRAF_512x512_MS.yaml)
  - Size: 1,282 bytes
  - SHA-256: `027dc3b0933b293e216ae8b59d147afddcbf88f85c4f3a4a656750bfc91de2d3`

### 4.3 CPU Compatibility Adapter & Execution Stub

FCSGG originally required compiling custom Detectron2 C++/CUDA extensions. On this Windows CPU host, compilation is bypassed through [`experiments/FCSGG/fcsgg_adapter.py`](file:///d:/Minor/experiments/FCSGG/fcsgg_adapter.py):
1. **Dynamic C-Extension Stub**: Injects a synthetic `types.ModuleType("detectron2._C")` into `sys.modules`, replacing native NMS with torchvision's pure Python/PyTorch NMS.
2. **Pillow 10+ Resampling Alias**: Automatically aliases `PIL.Image.LINEAR = PIL.Image.BILINEAR` to prevent deprecation crashes during multiscale inference.

---

## 5. REACT++ Resources

### 5.1 Model Architecture (YOLO12m Single-Stage ONNX)

REACT++ eliminates the two-stage Faster R-CNN/YOLOv8 + Transformer pipeline in favor of a single-stage network:
- **Backbone**: Ultralytics YOLO12m.
- **Relational Head**: Detection-Anchored Multi-Scale Pooling (DAMP) with an AIFI scene context branch.
- **Export Format**: Standalone ONNX graph with input `[batch_size, 3, 640, 640]` and compound outputs `boxes` (`[num_objects, 6]`) and `rels` (`[num_rels, 5]`).

### 5.2 Dataset Exports & Transfer Strategies

1. **REACT++ on PSG**:
   - Path: [`d:\Minor\checkpoints\PSG\react++_yolo12m\yolo12m\react_pp_yolo12m.onnx`](file:///d:/Minor/checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx)
   - Size: 116,791,801 bytes (111.38 MB)
   - SHA-256: `38c4bdbb7e2a077131ffcbf57af726eb30574318e2344d5b4bfc1d3a48f5b2fe`
   - Upstream URL: Hugging Face `https://huggingface.co/maelic/REACTPlusPlus_PSG` (`yolo12m/react_pp_yolo12m.onnx`).
   - Classes: 133 OpenPSG object categories, 56 predicate categories.
   - Verification: **PASS** (`onnxruntime` verified).
   - Hydra Config: [`model_repos/SGG-Benchmark/configs/hydra/PSG/REACT++.yaml`](file:///d:/Minor/model_repos/SGG-Benchmark/configs/hydra/PSG/REACT++.yaml) (4,469 bytes, SHA-256: `f36bdb04...`).

2. **REACT++ on VG150**:
   - Path: [`d:\Minor\checkpoints\REACT\react_pp_vg150_yolo12m.onnx`](file:///d:/Minor/checkpoints/REACT/react_pp_vg150_yolo12m.onnx)
   - Size: 116,984,567 bytes (111.57 MB)
   - SHA-256: `ce01616b902271da224c61ff85f9914f8efb12c8894e4aa7bad42ebe9480c9f2`
   - Upstream URL: Hugging Face `https://huggingface.co/maelic/REACTPlusPlus_VG150` (`yolo12m/model.onnx`).
   - Classes: 150 VG object categories, 50 predicate categories.
   - Verification: **PASS** (`onnxruntime` verified).
   - Hydra Config: [`model_repos/SGG-Benchmark/configs/hydra/VG/REACT++.yaml`](file:///d:/Minor/model_repos/SGG-Benchmark/configs/hydra/VG/REACT++.yaml) (4,954 bytes, SHA-256: `d505b5b8...`).

3. **REACT++ on IndoorVG & OpenImages V6**:
   - There are **no separate native checkpoints** trained on IndoorVG or OpenImages for REACT++.
   - Runs RUN-07 (IndoorVG) and RUN-08 (OpenImages) use `react_pp_vg150_yolo12m.onnx` transferred via our 3-Tier object projection and canonical predicate routing.

---

## 6. Fast Contextual SGG (C-SGG) Resources

1. **YOLOv5l Object Detector**:
   - Path: [`d:\Minor\checkpoints\FastContextualSGG\yolol_object_test_28.pt`](file:///d:/Minor/checkpoints/FastContextualSGG/yolol_object_test_28.pt)
   - Size: 376,488,983 bytes (359.05 MB)
   - SHA-256: `f657e21917f10d2d70da5e3f1b02d3148dc8589ac74ba6e7d8c2fc5e2aaac257`
   - Provenance: Google Drive `https://drive.google.com/uc?id=1f77tyIHTyDyRBupzA7vz9S82oc7DZ8E3` (downloaded by [`download_detector.py`](file:///d:/Minor/.cache/FastContextualSGG/download_detector.py)).
   - Verification: **PASS** (Loads via YOLOv5 `models.yolo.Model`; 151 classes).

2. **Context Knowledge Network (CKN) Multi-Layer Perceptrons**:
   - VG150 CKN Head: [`model_repos/fast-context-scene-graph-generation/VG_dataset/ckn/CKN+EPBS.pt`](file:///d:/Minor/model_repos/fast-context-scene-graph-generation/VG_dataset/ckn/CKN+EPBS.pt) (828,503 bytes, SHA-256: `ffddecd41adf7881...`)
   - PSG CKN Head: [`model_repos/fast-context-scene-graph-generation/PSG_dataset/ckn/2619.pt`](file:///d:/Minor/model_repos/fast-context-scene-graph-generation/PSG_dataset/ckn/2619.pt) (840,791 bytes, SHA-256: `3a93b2948c7406c7...`)
   - OpenImages CKN Head: [`model_repos/fast-context-scene-graph-generation/OID_dataset/ckn/ckn_oid_PBS.pt`](file:///d:/Minor/model_repos/fast-context-scene-graph-generation/OID_dataset/ckn/ckn_oid_PBS.pt) (787,415 bytes, SHA-256: `c106480f718cf6d8...`)
   - Upstream Archive: [`d:\Minor_2\repo.zip`](file:///d:/Minor_2/repo.zip) (79,345,723 bytes, SHA-256: `417bc9b052713533...`).

---

## 7. Download History, Shell Logs & Source Provenance

1. **PowerShell ReadLine History**:
   - Inspected `C:\Users\Lenovo\AppData\Roaming\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt` (116 KB, 2,106 command lines).
   - Confirmed explicit installation of CPU wheels: `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
   - No direct `wget`, `curl`, or `huggingface-cli` download commands for GloVe were found.
2. **Browser Download Databases**:
   - Checked Chrome and Edge SQLite download databases (`...\Google\Chrome\User Data\Default\History` and `...\Microsoft\Edge\User Data\Default\History`). Zero records existed for GloVe or SGG weight downloads.
3. **Repository Download Scripts & Extraction Logs**:
   - [`d:\Minor\.cache\FastContextualSGG\download_detector.py`](file:///d:/Minor/.cache/FastContextualSGG/download_detector.py): Implements direct programmatic download of `yolol_object_test_28.pt` from Google Drive via direct token extraction.
   - [`d:\Minor\.cache\FastContextualSGG\extract_glove.py`](file:///d:/Minor/.cache/FastContextualSGG/extract_glove.py): Streamed `https://huggingface.co/JeremiahZ/glove/resolve/main/glove.6B.50d.txt` to parse target words into `.cache/FastContextualSGG/glove_extracted_50d.pt`.
   - Checkpoint logs [`checkpoints/REACT/react_VG150/log.txt`](file:///d:/Minor/checkpoints/REACT/react_VG150/log.txt) and [`react_PSG/log.txt`](file:///d:/Minor/checkpoints/REACT/react_PSG/react_PSG/log.txt): Recorded training runs executed on upstream clusters by author Maelic Neau (`/home/maelic/glove`).

---

## 8. Master Resource Audit Matrix

### Category 1: Files Physically Present on Disk

| Resource | Found? | Exact Path | Size | Source URL / Provenance | SHA-256 Digest | Verified? |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: |
| REACT v1 VG150 Checkpoint | Yes | `d:\Minor\checkpoints\REACT\react_VG150\best_model_epoch_9.pth` | 225.05 MB | `https://github.com/Maelic/REACT` (author repo) | `8bf598afdb2ccef9e5b6f54dc1fe0fa70b46a4bfceecfb02f693f2e82ad503bf` | **YES** |
| REACT v1 VG150 Archive | Yes | `d:\Minor\checkpoints\REACT\react_VG150.zip` | 175.19 MB | `https://github.com/Maelic/REACT` | `e6c9082bdaf1614b6e3345df00236d6177c6212290023456dd005397c4254b4e` | **YES** |
| REACT v1 PSG Checkpoint | Yes | `d:\Minor\checkpoints\REACT\react_PSG\react_PSG\best_model_epoch_11.pth` | 225.00 MB | `https://github.com/Maelic/REACT` (author repo) | `0f27511e66f77f6d3eac4c09f5ccb06a61e702547032a2eddb1b601fe26ff03d` | **YES** |
| REACT v1 PSG Archive | Yes | `d:\Minor\checkpoints\REACT\react_PSG.zip` | 191.45 MB | `https://github.com/Maelic/REACT` | `3c0f92adc12bd38c95da358052a6f1d20fd5c3d09d4b9da79d578e3b5e628920` | **YES** |
| REACT v1 IndoorVG ONNX | Yes | `d:\Minor\checkpoints\REACT\react_indoorvg_yolov8m.onnx` | 132.45 MB | `https://huggingface.co/maelic/REACTPlusPlus_IndoorVG` | `5188b578aa155e3584d95b0e3d87a5bbd80ae2464f6d1f3e380d9b7ad872c7e8` | **YES** |
| REACT++ PSG ONNX | Yes | `d:\Minor\checkpoints\PSG\react++_yolo12m\yolo12m\react_pp_yolo12m.onnx` | 111.38 MB | `https://huggingface.co/maelic/REACTPlusPlus_PSG` | `38c4bdbb7e2a077131ffcbf57af726eb30574318e2344d5b4bfc1d3a48f5b2fe` | **YES** |
| REACT++ VG150 ONNX | Yes | `d:\Minor\checkpoints\REACT\react_pp_vg150_yolo12m.onnx` | 111.57 MB | `https://huggingface.co/maelic/REACTPlusPlus_VG150` | `ce01616b902271da224c61ff85f9914f8efb12c8894e4aa7bad42ebe9480c9f2` | **YES** |
| FCSGG HRNetW32-1S Checkpoint | Yes | `d:\Minor\checkpoints\FCSGG\HRNetW32-1S.pth` | 362.41 MB | Google Drive `1IpSDAZQW8irJyfWW5jep9fBVPRZcsTzl` | `41aa7639bac0793ef6ad04b860b2d441a32523cf0f69b7590b2eda261638c084` | **YES** |
| C-SGG YOLOv5l Detector | Yes | `d:\Minor\checkpoints\FastContextualSGG\yolol_object_test_28.pt` | 359.05 MB | Google Drive `1f77tyIHTyDyRBupzA7vz9S82oc7DZ8E3` | `f657e21917f10d2d70da5e3f1b02d3148dc8589ac74ba6e7d8c2fc5e2aaac257` | **YES** |
| C-SGG Upstream Repo Archive | Yes | `d:\Minor_2\repo.zip` | 75.67 MB | Jin et al. CVPR 2023 official repository | `417bc9b05271353389f81615ad60add327b2c2e0de921abc6907bc94bc0ab08c` | **YES** |

### Category 2: Files Available Only in Caches / Subsets

| Resource | Found? | Exact Path | Size | Source URL / Provenance | SHA-256 Digest | Verified? |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: |
| REACT GloVe 200d Stub | Yes | `d:\Minor\.cache\REACT\glove\glove.6B.200d.pt` | 2,387 B | Locally generated zero-stub to bypass 862 MB download | `0c401134071e39c1b95de7420863a340dcf38c8fb0f53b4e205ad0694c4829c3` | **YES (STUB)** |
| C-SGG GloVe 50d Extracted Subset | Yes | `d:\Minor\.cache\FastContextualSGG\glove_extracted_50d.pt` | 101.48 KB | Streamed from HF `JeremiahZ/glove` | `e44525ce7f09cad236498b4e097ec48d9ba146e747b7cccef59f6e75fe760da3` | **YES (SUBSET)** |
| C-SGG VG Word Vector Tensor | Yes | `d:\Minor\.cache\FastContextualSGG\vg_word_vec_list.pt` | 31.10 KB | Generated from `glove_extracted_50d.pt` (151 $\times$ 50) | `a3bc639414c53b20d2b48a1eff8efb55a46ab05adbc59137f625c8d085ccd47d` | **YES** |
| C-SGG PSG Word Vector Tensor | Yes | `d:\Minor\.cache\FastContextualSGG\psg_word_vec_list.pt` | 27.73 KB | Generated from `glove_extracted_50d.pt` (134 $\times$ 50) | `dd42ac400e8e6c41f3cfbac235848f03f456c58494a26ed2fe5824bffc44be22` | **YES** |

### Category 3: Files Downloadable from Verified Upstream Sources

| Resource | Found? | Exact Path | Size | Source URL / Provenance | Upstream Verification |
| :--- | :---: | :--- | :---: | :--- | :--- |
| Official `glove.6B.zip` | No | *Not on disk* | ~862 MB | `http://nlp.stanford.edu/data/glove.6B.zip` | Stanford NLP Official Archive |
| REACT v1 IndoorVG PyTorch (`best_model.pth`) | No | *Not on disk* | ~235 MB | `https://huggingface.co/maelic/REACTPlusPlus_IndoorVG/resolve/main/yolov8m/best_model.pth` | Official Hugging Face Model Hub |

### Category 4: Genuinely Missing / Non-Existent Files

| Resource | Found? | Exact Path | Size | Description / Reason |
| :--- | :---: | :--- | :---: | :--- |
| Complete `glove.6B.200d.txt` | No | *Not on disk* | ~693 MB | Uncompressed text format from `glove.6B.zip` |
| Complete `glove.6B.50d.txt` | No | *Not on disk* | ~171 MB | Uncompressed text format from `glove.6B.zip` |
| FCSGG PSG-Trained Checkpoint | No | *Not on disk* | — | Does not exist upstream (model trained exclusively on VG150) |
| FCSGG IndoorVG-Trained Checkpoint | No | *Not on disk* | — | Does not exist upstream (model trained exclusively on VG150) |

---

## 9. Handoff, Packaging & Safe Transfer Guide

### 9.1 Essential Transfer Payload

To reproduce the benchmark on a second computer without downloading external files from Google Drive or Hugging Face, transfer the following files (total payload: **~1.52 GB**):

1. **Pre-packaged Model Archives**:
   - `d:\Minor\checkpoints\REACT\react_VG150.zip` (175.19 MB)
   - `d:\Minor\checkpoints\REACT\react_PSG.zip` (191.45 MB)
   - `d:\Minor_2\repo.zip` (75.67 MB)
2. **Standalone ONNX Models**:
   - `d:\Minor\checkpoints\PSG\react++_yolo12m\yolo12m\react_pp_yolo12m.onnx` (111.38 MB)
   - `d:\Minor\checkpoints\REACT\react_pp_vg150_yolo12m.onnx` (111.57 MB)
   - `d:\Minor\checkpoints\REACT\react_indoorvg_yolov8m.onnx` (132.45 MB)
3. **PyTorch Weight Checkpoints**:
   - `d:\Minor\checkpoints\FCSGG\HRNetW32-1S.pth` (362.41 MB)
   - `d:\Minor\checkpoints\FastContextualSGG\yolol_object_test_28.pt` (359.05 MB)
4. **Pre-Extracted GloVe Word Vectors**:
   - `d:\Minor\.cache\FastContextualSGG\glove_extracted_50d.pt` (101.48 KB)
   - `d:\Minor\.cache\FastContextualSGG\vg_word_vec_list.pt` (31.10 KB)
   - `d:\Minor\.cache\FastContextualSGG\psg_word_vec_list.pt` (27.73 KB)

### 9.2 Transfer Methods

#### Method A: Local Network Transfer (`scp` / `rsync`)
```bash
# Transfer checkpoints directory
scp -r d:/Minor/checkpoints user@target_ip:/path/to/minor_project/

# Transfer GloVe semantic cache
scp -r d:/Minor/.cache/FastContextualSGG/*.pt user@target_ip:/path/to/minor_project/.cache/FastContextualSGG/
```

#### Method B: Unified Archive Bundle (External USB or Shared Drive)
Run this command from `d:\Minor` to generate a single compressed archive:
```powershell
tar -czvf sgg_checkpoints_bundle.tar.gz -C d:/Minor checkpoints .cache/FastContextualSGG
```
Copy `sgg_checkpoints_bundle.tar.gz` to the target computer and unpack it at the repository root:
```powershell
tar -xzvf sgg_checkpoints_bundle.tar.gz
```
