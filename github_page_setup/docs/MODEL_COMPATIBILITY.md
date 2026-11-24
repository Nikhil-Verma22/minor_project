# Model Compatibility Report

**Project:** Object Retrieval in Dynamic Workspace Environments  
**Primary SGG Model:** REACT++ (YOLOv12n PSG ONNX export)  
**Fallback VLM:** SmolVLM-500M-Instruct (ExecuTorch WebGPU export)  
**Date of Audit:** 2026-11-20  

---

## 1. Resource & Checkpoint Inventory

| Component | Asset Identifier / Repository | File Path / Source URL | File Size | Checksum (SHA-256) | Status |
|---|---|---|---|---|---|
| **SGG Checkpoint (Primary)** | `maelic/REACTPlusPlus_PSG` (yolo12n) | `public/models/react_pp_yolo12n.onnx` | 45,362,728 bytes (~45.4 MB) | `5e8f06c6094629c09dba354e8c631f6617dc602d82783e52cad48d8d903a1fbe` | Verified & Loaded |
| **SGG Quality Variant (Optional)** | `maelic/REACTPlusPlus_PSG` (yolo12m) | `d:\Minor\checkpoints\PSG\react++_yolo12m\yolo12m\react_pp_yolo12m.onnx` | 116,793,607 bytes (~116.8 MB) | Verified on disk | Available for manual switch |
| **ONNX Runtime Web WASM** | `onnxruntime-web@1.31.1` | `public/ort/*.wasm` (simd-threaded, jsep, jspi) | ~91 MB total (4 binaries) | Locally mirrored | Verified & Functional |
| **VLM Model Package** | `executorch-community/SmolVLM-500M-Instruct-ExecuTorch-WebGPU` | Hugging Face Hub (Lazy-loaded via Cache API / OPFS) | 518.2 MB total package (Decoder: 366.9MB, Vision: 103.8MB, Embed: 47.5MB) | Verified via HF API manifest | Meets < 2 GB budget |

---

## 2. REACT++ ONNX Graph and Tensor Contract

### Tensor Signatures
- **Input:**
  - Name: `input`
  - Shape: `[1, 3, 640, 640]`
  - Type: `float32`
  - Normalization: Scale to `[0.0, 1.0]` by dividing pixel values by `255.0` (RGB order).
- **Outputs:**
  1. `boxes`:
     - Shape: `[N, 6]` (where $N$ is variable detected boxes count).
     - Type: `float32`
     - Columns: `[x1, y1, x2, y2, class_idx_1based, score]`
  2. `rels`:
     - Shape: `[M, 5]` (where $M$ is predicted relations count).
     - Type: `float32`
     - Columns: `[subj_box_idx, obj_box_idx, rel_class_0based, triplet_score, rel_score]`

### Letterbox Preprocessing & Un-letterbox Coordinate Scaling
- Target input size: $640 \times 640$, background padding fill color: RGB `(114, 114, 114)`.
- Given original image of width $W$ and height $H$:
  $$r = \min\left(\frac{640}{H}, \frac{640}{W}\right)$$
  $$N_W = \text{round}(W \cdot r), \quad N_H = \text{round}(H \cdot r)$$
  $$\text{pad}_x = \left\lfloor\frac{640 - N_W}{2}\right\rfloor, \quad \text{pad}_y = \left\lfloor\frac{640 - N_H}{2}\right\rfloor$$
- Bounding box reconstruction:
  $$x_1 = \frac{x_{1,\text{raw}} - \text{pad}_x}{r}, \quad y_1 = \frac{y_{1,\text{raw}} - \text{pad}_y}{r}$$
  $$x_2 = \frac{x_{2,\text{raw}} - \text{pad}_x}{r}, \quad y_2 = \frac{y_{2,\text{raw}} - \text{pad}_y}{r}$$

### Score Calculation Contract
- Column 3 of `rels` is the geometric triplet score: $(s_{\text{subj}} \cdot s_{\text{obj}} \cdot s_{\text{rel}})^{1/3}$.
- Column 4 of `rels` is the pure predicate confidence score $s_{\text{rel}}$.
- Relations are filtered using the triplet score threshold (default: $0.05$ - $0.10$).

### Metadata & Indexing Contract
- `obj_classes` embedded JSON array contains 133 classes (background not included). Class index `class_idx_1based` from `boxes[:, 4]` is 1-based, meaning class name is `obj_classes[class_idx_1based - 1]`.
- `rel_classes` embedded JSON array contains 57 classes starting with `__background__` at index 0. `rel_class_0based` from `rels[:, 2]` maps directly to `rel_classes[rel_class_0based]`.

---

## 3. Browser & Local Execution Verification

### Numerical Consistency Test (Golden Python vs Browser ORT-Web)
Tested on evaluation image `107907.jpg` ($640 \times 480$):
- **Python Standalone ONNX Runtime:**
  - Boxes: `(13, 6)`, Relations: `(57, 5)`
  - Box 0: `tree` ($s=0.883$) at $[0.6, 1.6, 640.0, 360.3]$
  - Box 1: `grass` ($s=0.784$) at $[-0.4, 284.1, 640.2, 480.1]$
- **onnxruntime-web (WASM SIMD Multi-threaded):**
  - Boxes: `[13, 6]`, Relations: `[57, 5]`
  - Box 0: class 117 (`tree`), score `0.883`, raw $[0.6, 81.6, 640.0, 440.3] \rightarrow [0.6, 1.6, 640.0, 360.3]$ after unpadding ($\text{pad}_y = 80$).
  - Box 1: class 126 (`grass`), score `0.784`, raw $[-0.4, 364.1, 640.2, 560.1] \rightarrow [-0.4, 284.1, 640.2, 480.1]$.
  - **Verdict:** 100% exact numerical match across all boxes, scores, and relationship triplets.

### Browser Execution Providers Order
1. **Primary:** `webgpu` via ONNX Runtime Web. If supported by browser GPU context (`navigator.gpu`).
2. **Fallback 1:** `wasm` with SIMD and multi-threading via Web Workers. Provides instant, reliable offline inference (tested latency: ~200 ms for forward pass).
3. **Fallback 2:** Local Python HTTP sidecar service (`local-inference/server.py`) on `127.0.0.1:8000` using standard CPU/DirectML/CUDA execution if browser WebGPU/WASM are restricted by policy.

---

## 4. VLM Integration Feasibility (SmolVLM-500M)

- **Selected Model:** `executorch-community/SmolVLM-500M-Instruct-ExecuTorch-WebGPU`
- **Parameter Count:** 507 Million
- **Storage Footprint:**
  - `smolvlm_decoder_q8.pte`: 366.9 MB
  - `smolvlm_vision_q8.pte`: 103.8 MB
  - `embed_tokens.i8.bin`: 47.3 MB
  - `embed_tokens.scale.f32.bin`: 0.2 MB
  - Metadata / Tokenizer: ~0.5 MB
  - Total: **518.2 MB** (Strictly well below the 2.0 GB ceiling).
- **Execution Policy:** Lazy loading only. Model files are never requested until an NLQ query cannot be answered by structured scene graph and memory reasoning.
- **Caching:** Browser Cache API / OPFS (Origin Private File System) handles persistent caching to avoid redundant downloads.

---

## 5. Architectural Risks & Mitigations

1. **Vocabulary Gap:** Pretrained OpenPSG model recognizes 133 classes (COCO/OpenPSG subset) rather than all 39 custom workspace classes.
   - *Mitigation:* A robust `VocabularyAdapter` explicitly maps supported overlapping classes and safely marks out-of-vocabulary detections as `unmapped` while preserving the raw label.
2. **Memory Overload:** Loading both models in GPU memory on low-memory devices (e.g., 8 GB unified memory).
   - *Mitigation:* SGG model runs in a separate Web Worker; VLM is lazy-loaded and exposes explicit dispose/unload routines.
3. **Predicate Canonicalization:** VLM free-form output may propose non-canonical predicates.
   - *Mitigation:* All VLM facts are quarantined into a `vlmCandidates` table marked as `provisional`, requiring user verification before promotion to the canonical relationship store.
