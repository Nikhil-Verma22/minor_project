# Compatibility Report: Pretrained REACT Evaluation

**Research Project**: *Object Retrieval in Dynamic Workspace Environments*  
**Model Under Evaluation**: REACT (Relation-Enhanced Architecture with Context-guided Transformers, BMVC 2025)  
**Evaluation Harness Workspace**: `d:/Minor`  
**Evaluation Scope**: Visual Genome (VG150), Panoptic Scene Graph (PSG), IndoorVG, Open Images V6  

---

## 1. Executive Summary & Methodological Guardrails

Evaluating pretrained Scene Graph Generation (SGG) models across heterogeneous benchmarks requires strict methodological rigor. A common error in literature is forcing cross-dataset Scene Graph Detection (**SGDet**) under the premise that predicate mappings alone (such as projecting relation types onto a shared ontology) suffice for cross-domain evaluation.

### Core Compatibility Principles Enforced:
1. **Object Vocabulary Incompatibility**: SGDet requires joint localization, object classification, and relation classification. A model whose object detector was trained on $C_{\text{src}}$ object categories cannot meaningfully detect and classify objects in an evaluation dataset with disjoint or differently annotated categories $C_{\text{tgt}}$. Evaluating a PSG detector (133 classes) against VG150 (150 classes) or Open Images (601 classes) in SGDet mode yields artificial 0% recall for non-overlapping categories and distorts comparative metrics.
2. **Data Leakage & Training Exposure**: Verification of dataset image identifiers revealed that **100% of the 200 images in `IndoorVG_200` overlap with the Visual Genome (VG150) training split** (`LEAK_INDOOR_VGTRAIN`). Evaluating a VG150-pretrained model on IndoorVG measures training set memorization, not zero-shot or out-of-distribution transfer.
3. **Defensible Evaluation Mandate**: Only **methodologically defensible** model–dataset combinations are evaluated. Disjoint cross-dataset combinations are formally documented in the compatibility matrix as unsupported, preventing skewed benchmarks.

---

## 2. Checkpoint Provenance & Model Specifications

The official REACT models were sourced from the BMVC 2025 release repository (`https://github.com/Maelic/SGG-Benchmark`):

| Attribute | REACT-PSG | REACT-VG150 |
| :--- | :--- | :--- |
| **Release Tag** | `REACTv1` (BMVC 2025) | `REACTv1` (BMVC 2025) |
| **Download Source** | Google Drive (`1uxohHdeh4eZ-FG81DS-ooJZFcWEHd3uX`) | Google Drive (`1q7WAcJ9XS5ilt3Cf3ysBjwcz5CcaUzdJ`) |
| **Local Checkpoint** | `checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth` | `checkpoints/REACT/react_VG150/best_model_epoch_9.pth` |
| **Detector Architecture** | YOLOv8m (169 layers, 25.9M params) | YOLOv8m (169 layers, 25.9M params) |
| **Detector Output Classes** | 134 (133 PSG categories + 1 background) | 151 (150 VG categories + 1 background) |
| **Relation Head** | `REACTPredictor` (MLP head dim: 512) | `REACTPredictor` (MLP head dim: 512) |
| **Relation Output Classes**| 57 (56 PSG predicates + 1 background) | 51 (50 VG predicates + 1 background) |
| **Feature Extractor** | YOLO Multi-Scale RoI Pooling (scales: [0.0625]) | YOLO Multi-Scale RoI Pooling (scales: [0.0625]) |
| **Input Resolution** | $640 \times 640$ (letterbox padded) | $640 \times 640$ (letterbox padded) |
| **Supported Tasks** | SGDet, PredCls, SGCls | SGDet, PredCls, SGCls |

---

## 3. Dataset Characteristics & Vocabularies

Each frozen evaluation dataset contains a standardized 200-image subset:

| Dataset | Split | Images | Objects | Native Predicates | Normalized Ontology Views |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PSG_200** | Test | 200 | 133 classes (COCO Thing/Stuff) | 56 predicates | `strict` (6), `extended` (8), `tri_dataset` (20) |
| **VG150_200** | Test | 200 | 150 classes | 50 predicates | `strict` (6), `extended` (8), `tri_dataset` (20) |
| **IndoorVG_200** | Test | 200 | 84 indoor classes | 40 predicates | `strict` (6), `extended` (8), `tri_dataset` (20) |
| **OpenImages_200**| Test | 200 | 601 classes | 30 predicates | `strict` (6), `extended` (8), `tri_dataset` (20) |

---

## 4. Comprehensive Compatibility Matrix

The matrix below documents every combination of model checkpoint and evaluation dataset:

| Model Checkpoint | Target Dataset | Task | Annotation Compatibility | Valid Ontology Views | Training Exposure | Status | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **REACT-PSG** | **PSG_200** | **SGDet** | **Full (Native)** | `strict`, `extended`, `tri_dataset`, `native` | None (Test Split) | **SUPPORTED & EVALUATED** | Native benchmark combination. Full object and predicate alignment. |
| **REACT-VG150** | **VG150_200** | **SGDet** | **Full (Native)** | `strict`, `extended`, `tri_dataset`, `native` | None (Test Split) | **SUPPORTED & EVALUATED** | Native benchmark combination. Full object and predicate alignment. |
| **REACT-PSG** | **VG150_200** | SGDet | Incompatible | None | Cross-dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (133 vs 150). Detector cannot recognize VG150 classes. |
| **REACT-PSG** | **IndoorVG_200** | SGDet | Incompatible | None | Cross-dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (133 vs 84). |
| **REACT-PSG** | **OpenImages_200** | SGDet | Incompatible | None | Cross-dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (133 vs 601). Detector cannot recognize OpenImages classes. |
| **REACT-VG150** | **PSG_200** | SGDet | Incompatible | None | Cross-dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (150 vs 133). Detector cannot recognize COCO/PSG classes. |
| **REACT-VG150** | **IndoorVG_200** | SGDet | High Leakage | `strict`, `extended`, `tri_dataset` | **100% Contaminated (`LEAK_INDOOR_VGTRAIN`)** | **REJECTED (DATA LEAKAGE)** | 100% of images are in VG150 training set. Evaluating here measures memorization. |
| **REACT-VG150** | **OpenImages_200** | SGDet | Incompatible | None | Cross-dataset | **REJECTED (UNSUPPORTED)** | Disjoint object vocabularies (150 vs 601). Detector cannot recognize OpenImages classes. |

---

## 5. Ontology Views & Dynamic Projection Architecture

To evaluate cross-dataset predicate harmonization without corrupting the original ground truth annotations in `evaluation_sets/`, we implemented `experiments/REACT/ontology_projection.py`:

1. **Strict View (6 Core Predicates)**:
   - `eating`, `hanging from`, `holding`, `in`, `on`, `wearing`
   - Direct spatial and physical contact relationships universally present across all four benchmark datasets.
2. **Extended View (8 Predicates)**:
   - Strict predicates + `looking at`, `playing`
   - Incorporates high-frequency interactive and observational verbs.
3. **Tri-Dataset View (20 Core Predicates)**:
   - `against`, `at`, `attached to`, `carrying`, `eating`, `hanging from`, `has`, `holding`, `in`, `in front of`, `looking at`, `lying on`, `near`, `on`, `playing`, `riding`, `sitting on`, `standing on`, `under`, `wearing`
   - The standardized cross-dataset vocabulary spanning common human–object and object–object interactions.

### Projection Properties Enforced:
- **Directionality Invariance**: Endpoint roles (subject vs object) are strictly preserved during projection.
- **Unmapped Filtering**: Predicates not defined within the view have their logits zeroed out; they do not compete in Top-K ranking for that view.
- **Duplicate Prevention**: Matching uses set-union logic (`np.union1d`) ensuring that multiple predictions matching a single ground truth triplet only register a single recall hit.

---

## 6. Access & Repository Provenance Audit

- **Team Repository Access Issue**: An initial clone attempt to the internal evaluation repository (`https://github.com/Sshivanshkaushik/SGG-Benchmark`) returned HTTP 404 (`Repository not found`). In accordance with research integrity protocols, this was recorded rather than silently substituted.
- **Upstream Repository Audit**: The official upstream repository (`https://github.com/Maelic/SGG-Benchmark`) was inspected as reference material under `model_repos/SGG-Benchmark/`. No write, commit, or push operations were conducted.
- **Dataset Read-Only Integrity**: A SHA-256 manifest of all 851 files in `evaluation_sets/` was created prior to benchmark execution and stored in `.cache/REACT/eval_sets_sha256_pre.json`. Post-execution hash comparison guarantees zero alterations.
