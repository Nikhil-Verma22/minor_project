# Compatibility & Inspection Report: Pretrained FCSGG Evaluation

**Research Project**: *Object Retrieval in Dynamic Workspace Environments*  
**Model Under Evaluation**: FCSGG (Fully Convolutional Scene Graph Generation, CVPR 2021)  
**Authors**: Hengyue Liu, Ning Yan, Masood Mortazavi, Bir Bhanu (Futurewei Technologies Inc. / UC Riverside)  
**Official Repository**: `https://github.com/liuhengyue/fcsgg`  
**Evaluation Scope**: Visual Genome (VG150), Panoptic Scene Graph (PSG), IndoorVG, Open Images V6  
**Execution Runtime**: PyTorch 2.8.0 on CPU (`Intel(R) Core(TM) Ultra 5 125H`, 18 threads)  

---

## 1. Executive Summary & Architecture Inspection

This report documents the architectural, methodological, and vocabulary inspection of the **FCSGG** model prior to benchmark execution.

### Key Inspection Findings:

1. **Fully Convolutional RAF Architecture**:
   - Unlike two-stage Faster R-CNN pipelines (e.g. SGG-Benchmark) or Transformer-based relation heads (e.g. REACT), FCSGG reformulates scene graph generation as a **bottom-up center-based point and vector regression problem**.
   - **Object Detection**: Implemented via a CenterNet meta-architecture. Objects are predicted via class-specific center keypoint heatmaps, 2D center offset regressions, and box scale $(w, h)$ regressions.
   - **Relationship Modeling via Relation Affinity Fields (RAFs)**: Relationships are encoded as dense 2D vector fields pointing from subject center heatmaps to object center heatmaps across each predicate channel ($2 \times P$ output channels).
   - **Decoding**: Center pairs are scored by line-integral sampling along the predicted RAF vector connecting subject and object center locations, producing triplet scores without combinatorial RoI pair pooling.

2. **Native Vocabularies & Dataset Origin**:
   - The official FCSGG architecture is natively configured and trained on **Visual Genome (VG150)**:
     - **Object Classes**: 150 categories (1-indexed, corresponding to canonical VG150 entities).
     - **Predicate Classes**: 50 categories (1-indexed, corresponding to canonical VG150 relationships).
   - Supported inference mode: **SGDet** (joint object detection and relation discovery).

3. **Checkpoint Availability & Blocker Identification**:
   - No pre-existing FCSGG checkpoint exists in `checkpoints/FCSGG/` within the workspace.
   - In accordance with the project safety protocol (*"Use pretrained weights only. If required weights are missing, report their source and the blocker before downloading large files"*), the official Google Drive sources were identified from the upstream repository:
     - **Option 1 (Recommended / Primary)**: `HRNetW32-1S.pth` (362 MB, Google Drive ID: `1IpSDAZQW8irJyfWW5jep9fBVPRZcsTzl`, Config: `configs/FCSGG_HRNet_W32_2xDownRAF_512x512_MS.yaml`). Primary single-stage model reported in the paper.
     - **Option 2**: `ResNet50-4S-FPN×2.pth` (202 MB, Google Drive ID: `10rtVr16RO2hd_JyiaBh8eNuxHvyTENR1`, Config: `configs/FCSGG-Res50-BiFPN-P2P5-MultiscaleHead-MS.yaml`).
     - **Option 3**: `HRNetW48-5S-FPN×2.pth` (666 MB, Google Drive ID: `1T7zZ_Rq5_mBhf1G89ab4w_SKz39IucsG`, Config: `configs/FCSGG_HRNet_W48_DualHRFPN_5s_Fixsize_640x1024_MS.yaml`).
   - Action: Blocker reported to user to authorize downloading the verified official checkpoint.

4. **Runtime & Dependency Environment**:
   - Detectron2 submodule initialized at `model_repos/fcsgg/detectron2`.
   - Core dependencies `fvcore`, `iopath`, `cloudpickle` installed in Python environment.
   - Compatibility shims configured for Pillow 10+ (`Image.LINEAR` $\to$ `Image.BILINEAR`) and C++ extension mock (`detectron2._C`), allowing native PyTorch execution on Windows CPU.

---

## 2. Compatibility Matrix Across Benchmarks

| Model / Checkpoint | Target Dataset | Task | Native Annotation Compatibility | Valid Ontology Views | Training Exposure | Status | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FCSGG (VG150 Pretrained)** | **VG150_200** | **SGDet** | **Full (Native)** | `strict`, `extended`, `tri_dataset`, `native` | None (Test Split) | **SUPPORTED & TARGETED** | Native benchmark match. Full 150-object and 50-predicate alignment. |
| **FCSGG (VG150 Pretrained)** | **PSG_200** | SGDet | Incompatible | None | Disjoint Dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (150 VG classes vs 133 COCO classes). Detectors cannot detect target classes. |
| **FCSGG (VG150 Pretrained)** | **IndoorVG_200** | SGDet | Incompatible | None | **100% Training Contamination** | **REJECTED (DATA LEAKAGE)** | Disjoint object classes and 100% image overlap with the VG150 training split (`LEAK_INDOOR_VGTRAIN`). |
| **FCSGG (VG150 Pretrained)** | **OpenImages_200** | SGDet | Incompatible | None | Disjoint Dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (150 vs 601 classes). |

---

## 3. Ontology View Definitions for VG150

All predictions will be evaluated in SGDet mode across the four standardized ontology views established in Prompt 1:
1. **Strict Core (`strict`)**: 6 core spatial/contact predicates (`eating`, `hanging from`, `holding`, `in`, `on`, `wearing`).
2. **Extended Core (`extended`)**: 8 predicates (strict + `looking at`, `playing`).
3. **Tri-Dataset Core (`tri_dataset`)**: 20 standardized interaction predicates (`against`, `at`, `attached to`, `carrying`, `eating`, `hanging from`, `has`, `holding`, `in`, `in front of`, `looking at`, `lying on`, `near`, `on`, `playing`, `riding`, `sitting on`, `standing on`, `under`, `wearing`).
4. **Native VG150 (`native`)**: All 50 native predicates defined in Visual Genome.

---

## 4. Next Steps

1. User confirms authorized download of the official pretrained checkpoint (`HRNetW32-1S.pth` or `ResNet50-4S-FPN×2.pth`).
2. Save checkpoint to `checkpoints/FCSGG/`.
3. Validate FCSGG adapter with unit regression tests (coordinate unscaling, directionality, score ranking, ontology projection).
4. Run 1-image inference sanity check and 3-image pre-flight smoke test.
5. Execute full 200-image evaluation across all supported views.
