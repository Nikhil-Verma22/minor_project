# Compatibility & Inspection Report: Pretrained REACT++ Evaluation

**Research Project**: *Object Retrieval in Dynamic Workspace Environments*  
**Model Under Evaluation**: REACT++ (Relation-Enhanced Architecture with Context-guided Transformers ++, 2025)  
**Evaluation Scope**: Visual Genome (VG150), Panoptic Scene Graph (PSG), IndoorVG, Open Images V6  
**Checkpoint**: `checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx` (111.4 MB)  
**Execution Runtime**: ONNX Runtime 1.27.0 with `CPUExecutionProvider`  

---

## 1. Executive Summary & Checkpoint Inspection

This report documents the architectural, methodological, and vocabulary compatibility analysis for the **REACT++** model prior to full benchmark evaluation.

### Key Inspection Findings:
1. **Checkpoint Identification & Provenance**:
   - The candidate checkpoint `checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx` was initially missing from local workspace storage. Its official provenance was verified at Hugging Face (`maelic/REACTPlusPlus_PSG/yolo12m/react_pp_yolo12m.onnx`, 111.38 MB, HTTP 200).
   - Following user authorization, the exact official checkpoint was downloaded to `checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx`.
2. **End-to-End ONNX Graph Architecture**:
   - The ONNX model combines the YOLO12m detector, NMS proposal filtering, and the `REACTPlusPlusPredictor` relation head into a single unified computation graph.
   - **Input**: `input` tensor of shape `[1, 3, 640, 640]`, float32, normalized to $[0, 1]$, RGB format.
   - **Outputs**:
     - `boxes`: Shape `[N, 6]` containing `[x1, y1, x2, y2, label, score]` in $640 \times 640$ letterbox coordinates.
     - `rels`: Shape `[M, 5]` containing `[subj_idx, obj_idx, label, triplet_score, rel_score]`.
   - **Supported Task**: **SGDet only**. Because detection and proposal generation are baked into the static ONNX graph, the model does not accept ground-truth bounding boxes or labels as inputs; therefore, **PredCls and SGCls are not supported**.
3. **Native Vocabulary Verification**:
   - Metadata embedded in the ONNX model confirms:
     - **Object Classes**: 133 categories (COCO/PSG categories starting from `person`, 1-indexed).
     - **Relation Classes**: 57 categories (`__background__` at index 0, followed by the 56 native PSG relationship predicates).
4. **Native Decoding & Ontology Projection Safeguards**:
   - **Identified Risk**: Truncating or filtering raw ONNX relation outputs using the 6-predicate class count would discard all predicates with ID $\ge 7$ (e.g. `holding`, `carrying`, `driving on`).
   - **Implemented Solution**: Raw relation outputs are decoded using the full 57-class native vocabulary vector. The validated `OntologyProjector` then maps these native logits onto the target ontology views (`strict`, `extended`, `tri_dataset`, and `native`), preserving relation directionality and preventing duplicate ground-truth matching via set union (`np.union1d`).

---

## 2. Compatibility Matrix Across Benchmarks

| Model / Checkpoint | Evaluation Dataset | Supported Task | Native Annotation Compatibility | Valid Ontology Views | Training Exposure | Status | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REACT++ (YOLO12m)** | **PSG_200** | **SGDet** | **Full (Native)** | `strict`, `extended`, `tri_dataset`, `native` | None (Test Split) | **SUPPORTED & EVALUATED** | Native benchmark match. Both object and predicate vocabularies align with target dataset. |
| **REACT++ (YOLO12m)** | **VG150_200** | SGDet | Incompatible | None | Cross-dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (133 vs 150). The YOLO12m detector cannot classify VG150 categories. |
| **REACT++ (YOLO12m)** | **IndoorVG_200** | SGDet | Incompatible | None | **100% Training Contamination** | **REJECTED (DATA LEAKAGE)** | Disjoint object classes (133 vs 84) and 100% image overlap with the VG150 training split (`LEAK_INDOOR_VGTRAIN`). |
| **REACT++ (YOLO12m)** | **OpenImages_200** | SGDet | Incompatible | None | Cross-dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (133 vs 601). Detector cannot detect Open Images classes. |

---

## 3. Ontology View Definitions

The four evaluated views on PSG_200:
1. **Strict Core 4-Way (`strict`)**: 6 core spatial/contact predicates (`eating`, `hanging from`, `holding`, `in`, `on`, `wearing`).
2. **Extended Core 4-Way (`extended`)**: 8 predicates (strict + `looking at`, `playing`).
3. **Tri-Dataset Core (`tri_dataset`)**: 20 standardized interaction predicates (`against`, `at`, `attached to`, `carrying`, `eating`, `hanging from`, `has`, `holding`, `in`, `in front of`, `looking at`, `lying on`, `near`, `on`, `playing`, `riding`, `sitting on`, `standing on`, `under`, `wearing`).
4. **Native PSG (`native`)**: All 56 native predicates defined in the Panoptic Scene Graph benchmark.

---

## 4. Evaluator Validation & Pre-Flight Smoke Test

- **Unit Test Suite**: [test_reactpp_evaluator.py](file:///d:/Minor/experiments/REACTPP/test_reactpp_evaluator.py) executed and passed all 4 regression tests:
  - Native predicate decoding without 6-class truncation (`OK`).
  - Correct projection of native logits to `strict`, `extended`, and `tri_dataset` views (`OK`).
  - Relation endpoint directionality preservation (`OK`).
  - Formal compatibility assertion rejecting cross-dataset SGDet (`OK`).
- **Pre-Flight Smoke Test**: Executed across 3 test images:
  - Forward latency: $495.6$ ms / image.
  - End-to-end latency: $511.7$ ms / image.
  - Detection mAP@50: **0.7132**.
  - Confirmed all metrics, CSV, and JSON outputs before launching full 200-image evaluation.
