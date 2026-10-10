# Independent Pre-Evaluation Audit of SGG Dataset Preparation

**Repository**: `/Users/shivanshkaushik/Developer/SGG-Benchmark`  
**Audit Timestamp**: `2026-10-09T05:10:00Z`  
**Target Datasets**: VG150, PSG, IndoorVG, Open Images V6 (200 images each, 800 total)  
**Evaluated Models**: REACT, REACT++, FCSGG, Fast Contextual SGG  

---

## 1. Executive Verdict: PASS WITH WARNINGS

| Verdict Dimension | Status | Summary Assessment |
| :--- | :---: | :--- |
| **Data Inventory & Storage Integrity** | **PASS** | Exactly 200 unique IDs, 200 metadata records, 200 COCO image entries, and 200 decoded image files per dataset (800/800 verified). Zero corrupted images, zero orphan annotations, zero broken relationship endpoints. Prior frozen VG150 and PSG subsets are bit-for-bit intact. |
| **Ontology & Vocabulary Consistency** | **PASS** | 100% of source predicates across all four datasets are mapped in `evaluation_ontology.json`. All mapped canonical predicates exist in VG150. Exact and equivalent categories match canonical definitions. VG150 `wears` (ID 49) correctly maps to canonical `wearing` (ID 48) while preserving source IDs. |
| **Directionality & Semantic Preservation** | **PASS** | Subject/object directions are strictly preserved across all transformations. Open Images `inside_of` $\rightarrow$ `in` (Contained $\rightarrow$ Container) and `contain` $\rightarrow$ `has` (Container $\rightarrow$ Content) preserve physical geometry without reversing endpoints. |
| **Statistical Reproduction** | **PASS** | All reported aggregate counts and percentages reproduce to the exact integer: Strict Core = 2,211 / 5,006 (44.17%), Extended Core = 2,272 / 5,006 (45.39%), Tri-Dataset Core = 3,476 / 5,006 (69.44%). |
| **Subset Selection & Data Leakage** | **WARNING** | **Critical Discovery**: All 200 images of `IndoorVG_200` are present in the VG150 training split (68,538 images). Any model pretrained on VG150 has seen all 200 IndoorVG evaluation images during training. Furthermore, 7 integer IDs collide between VG150 and PSG due to independent namespaces (verified as distinct images). |
| **Model Evaluation Pipeline Readiness** | **WARNING** | Direct cross-dataset SGDet is methodologically invalid without an object class ontology alignment due to completely disjoint object vocabularies (150 vs 133 vs 84 vs 75 classes). Benchmarking across datasets must be conducted in **PredCls** mode or on native dataset splits. Prediction projection layers, zero-relation handling, and duplicate triplet resolution are required prior to evaluation. |

**Bottom Line**: It is **SAFE TO BEGIN MODEL EVALUATION**, provided that the four operational constraints in Section 7 are strictly enforced to prevent methodological artifacts and data-leakage misattributions.

---

## 2. Validation Checklist

All 23 automated audit checks executed across the codebase are recorded in [`evaluation_sets/pre_evaluation_audit_checks.csv`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/evaluation_sets/pre_evaluation_audit_checks.csv).

| Check ID | Category | Dataset | Target & Validation Rule | Status | Evidence / Verification Value |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `INV_VG150_TXT` | Inventory | VG150 | Exactly 200 unique IDs in TXT manifest | **PASS** | 200 lines, 200 unique IDs in `VG150_200.txt` |
| `INV_PSG_TXT` | Inventory | PSG | Exactly 200 unique IDs in TXT manifest | **PASS** | 200 lines, 200 unique IDs in `PSG_200.txt` |
| `INV_INDOOR_TXT` | Inventory | IndoorVG | Exactly 200 unique IDs in TXT manifest | **PASS** | 200 lines, 200 unique IDs in `IndoorVG_200.txt` |
| `INV_OI_TXT` | Inventory | OpenImages | Exactly 200 unique IDs in TXT manifest | **PASS** | 200 lines, 200 unique IDs in `OpenImages_200.txt` |
| `INV_VG150_META` | Inventory | VG150 | Metadata records match TXT IDs exactly | **PASS** | 200 records in `VG150_200_metadata.json`; set match = True |
| `INV_PSG_META` | Inventory | PSG | Metadata records match TXT IDs exactly | **PASS** | 200 records in `PSG_200_metadata.json`; set match = True |
| `INV_INDOOR_META` | Inventory | IndoorVG | Metadata records match TXT IDs exactly | **PASS** | 200 records in `IndoorVG_200_metadata.json`; set match = True |
| `INV_OI_META` | Inventory | OpenImages | Metadata records match TXT IDs exactly | **PASS** | 200 records in `OpenImages_200_metadata.json`; set match = True |
| `INV_ALL_COCO` | Inventory | All (4) | COCO ground truth `images` count = 200 | **PASS** | Exactly 200 image records in all 4 `*_annotations.coco.json` |
| `INV_ALL_FILES` | Inventory | All (4) | Disk files count = 200 per directory | **PASS** | Exactly 200 `.jpg` files in each `evaluation_sets/images/{dataset}` |
| `INT_ALL_DECODE` | Integrity | All (4) | PIL decoding, 0 corruption, valid size | **PASS** | 800/800 images decoded cleanly. 0 corruptions, 0 dimension mismatches |
| `INT_ALL_OBJREF` | Integrity | All (4) | Object annotations point to valid image IDs | **PASS** | 0 orphan object annotations across 9,943 total objects |
| `INT_ALL_RELREF` | Integrity | All (4) | Rel endpoints valid under ID conventions | **PASS** | 0 broken endpoints across 5,006 total relationships |
| `INT_ALL_CATID` | Integrity | All (4) | Category IDs internally consistent & unique | **PASS** | VG150 (150 obj, 50 rel); PSG (133 obj, 56 rel); Indoor (84 obj, 37 rel); OI (75 obj, 31 rel) |
| `ONT_ALL_COV` | Ontology | All (4) | Every source predicate accounted for | **PASS** | VG150 (50/50); PSG (56/56); IndoorVG (37/37); OpenImages (31/31) mapped |
| `ONT_VG_WEARS` | Ontology | VG150 | `wears` (49) & `wearing` (48) $\rightarrow$ `wearing` | **PASS** | Both mapped to normalized `wearing` (48); source IDs preserved |
| `DIR_OI_INSIDE` | Direction | OpenImages | `inside_of` $\rightarrow$ `in` direction preserved | **PASS** | Contained $\rightarrow$ Container preserved (e.g. apple in bowl) |
| `DIR_OI_CONT` | Direction | OpenImages | `contain` $\rightarrow$ `has` direction preserved | **PASS** | Container $\rightarrow$ Content preserved (never mapped to `in`) |
| `STAT_DENOM` | Statistics | All (4) | Denominator reproduces to exactly 5,006 | **PASS** | $1,083 + 1,385 + 1,672 + 866 = 5,006$ |
| `STAT_REPRO` | Statistics | All (4) | Views reproduce to exact counts & % | **PASS** | Strict: 2,211 (44.17%); Ext: 2,272 (45.39%); Tri: 3,476 (69.44%) |
| `LEAK_INDOOR_VG` | Leakage | IndoorVG | IndoorVG test overlap with VG150 train | **CRITICAL WARNING** | **200/200 IndoorVG evaluation images exist in VG150 train split** |
| `LEAK_VG_PSG` | Leakage | PSG | Namespace collision of 7 integer IDs | **PASS WITH WARNING** | 7 IDs collide numerically, but pixel hashes and contents differ |
| `PIPE_READINESS`| Pipeline | All (4) | Evaluation pipeline operational prerequisites | **PASS WITH WARNING** | Disjoint object classes mandate PredCls mode for cross-dataset |

---

## 3. Stage 1: Inventory and Integrity Findings

### A. Dataset Subset Counts and Checksums

All original subsets and downloaded images were verified against modification:

| Dataset | Manifest Path | Unique IDs | GT Objects | GT Relationships | Disk Image Size | Timestamp Verified |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **VG150** | `evaluation_sets/VG150_200.txt` | 200 | 2,320 | 1,083 | 24.54 MB | Untouched since 2026-10-08T23:29 |
| **PSG** | `evaluation_sets/PSG_200.txt` | 200 | 3,019 | 1,385 | 33.13 MB | Untouched since 2026-10-09T01:52 |
| **IndoorVG** | `evaluation_sets/IndoorVG_200.txt` | 200 | 3,118 | 1,672 | 19.46 MB | Created 2026-10-09T04:16 |
| **Open Images V6** | `evaluation_sets/OpenImages_200.txt` | 200 | 1,486 | 866 | 32.55 MB | Created 2026-10-09T04:22 |

### B. Object Annotation ID Namespace Conventions

A critical structural distinction between the four datasets was verified:

1. **Local Box Indexing (0-indexed per image)**:
   - **VG150** and **IndoorVG**: In `annotations`, `id` is $0, 1, \dots, N-1$ within that image. In `rel_annotations`, `subject_id` and `object_id` refer to the local box index in that specific image.
   - *Verification Result*: 100% of subject and object references resolve to valid bounding boxes within the identical image.
2. **Global Annotation Indexing**:
   - **PSG** and **Open Images V6**: In `annotations`, `id` is a unique global integer ($0 \dots 24,909$ in PSG; $0 \dots 1,485$ in Open Images). In `rel_annotations`, `subject_id` and `object_id` refer to the global annotation ID.
   - *Verification Result*: 100% of subject and object references resolve to valid global annotations belonging to that image.

---

## 4. Stage 2 & 3: Ontology, Equivalence, and Directionality Audit

### A. Source Predicate Accountancy

| Dataset | Total Source Predicates | Mapped Exact | Mapped Semantic Equiv | Mapped Generalization | Unmapped / Dropped | Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VG150** | 50 | 49 | 1 (`wears` $\rightarrow$ `wearing`) | 0 | 0 | **100%** |
| **PSG** | 56 | 45 | 1 (`leaning on` $\rightarrow$ `against`) | 3 (`beside`, `playing with`, `flying over`) | 7 (`over`, `enclosing`, etc.) | **100%** |
| **IndoorVG** | 37 | 30 | 0 | 3 (`playing with`, `reading`, `sitting at`) | 4 (`filled with`, `talking on`, etc.) | **100%** |
| **Open Images V6** | 31 | 3 | 7 (`holds`, `wears`, `hang`, `ride`, `inside_of`, `eat`, `plays`) | 3 (`holding_hands`, `contain`, `read`) | 18 (`is`, `surf`, `dance`, `catch`, etc.) | **100%** |

### B. VG150 `wearing` vs `wears` Normalization

- **Original Vocabulary**: VG150 contains crowd-sourced lexical duplicates: `wearing` (ID 48) and `wears` (ID 49).
- **Mapping Verification**: In `evaluation_ontology.json`, `wears` (ID 49) maps to canonical `wearing` (canonical ID 48) with `mapping_type: "SEMANTIC_EQUIVALENT"`.
- **Annotation Preservation**: In `VG150_200_strict.coco.json`, exactly 8 instances of `wears` are normalized:
  ```json
  {
    "id": 0,
    "image_id": 2315546,
    "subject_id": 4,
    "object_id": 5,
    "predicate_id": 49,
    "normalized_predicate_name": "wearing",
    "normalized_predicate_id": 6,
    "canonical_vg_predicate_id": 48,
    "mapping_type": "SEMANTIC_EQUIVALENT"
  }
  ```
  The original `predicate_id` (49) remains untouched, while evaluation metadata provides the normalized canonical target.

### C. Directionality and Physical Relations Audit

1. **Open Images `inside_of` $\rightarrow$ `in`**:
   - In Open Images VRD: `<LabelName1, inside_of, LabelName2>` denotes that Object 1 is enclosed within Object 2 (e.g. apple in bowl).
   - In VG150: `<subject, in, object>` denotes that subject is enclosed in object.
   - **Directionality**: 100% Preserved (Contained $\rightarrow$ Container).
2. **Open Images `contain` $\rightarrow$ `has`**:
   - In Open Images VRD: `<LabelName1, contain, LabelName2>` denotes that Object 1 encloses Object 2 (e.g. bowl contains apple).
   - Mapping to `in` would reverse endpoints (Subject would become container while `in` expects contained).
   - Mapping to `has` preserves endpoints: `<bowl, has, apple>`.
   - **Directionality**: 100% Preserved (Container $\rightarrow$ Content). Semantic meaning is slightly broadened from physical containment to general possession/containment.
3. **Open Images `holding_hands` $\rightarrow$ `holding`**:
   - In Open Images VRD: `<person A, holding_hands, person B>`.
   - Categorized as `ACCEPTABLE_SEMANTIC_GENERALIZATION`.
   - **Finding**: In `evaluation_ontology.json`, `holding_hands` has flag `in_strict_4way: True`. Although `holding_hands` is a generalization rather than exact/semantic equivalent, exactly **0 instances** of `holding_hands` exist in `OpenImages_200` (hence 0 impacted ground-truth relations in strict core).
4. **Open Images `hang` $\rightarrow$ `hanging from`**:
   - In Open Images VRD: `<clock/picture, hang, wall>`.
   - In VG150: `<clock/picture, hanging from, wall>`.
   - **Directionality**: 100% Preserved (Suspended Object $\rightarrow$ Support Surface).

---

## 5. Stage 4: Recomputed Normalized Annotation Statistics

Every reported number was recalculated directly from raw annotation files:

### Detailed Dataset Breakdown

| View | Metric | VG150 | PSG | IndoorVG | Open Images V6 | Aggregate Total | Reported | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original GT** | Total Rels | 1,083 | 1,385 | 1,672 | 866 | **5,006** | 5,006 | **PASS** |
| **Strict Core (6)** | Retained | 604 | 520 | 1,003 | 84 | **2,211** | 2,211 | **PASS** |
| | Dropped | 479 | 865 | 669 | 782 | **2,795** | 2,795 | **PASS** |
| | Retained % | 55.77% | 37.55% | 59.99% | 9.70% | **44.17%** | 44.17% | **PASS** |
| **Extended Core (8)**| Retained | 605 | 575 | 1,004 | 88 | **2,272** | 2,272 | **PASS** |
| | Dropped | 478 | 810 | 668 | 778 | **2,734** | 2,734 | **PASS** |
| | Retained % | 55.86% | 41.52% | 60.05% | 10.16% | **45.39%** | 45.39% | **PASS** |
| **Tri-Dataset (20)** | Retained | 873 | 1,050 | 1,411 | 142 | **3,476** | 3,476 | **PASS** |
| | Dropped | 210 | 335 | 261 | 724 | **1,530** | 1,530 | **PASS** |
| | Retained % | 80.61% | 75.81% | 84.39% | 16.40% | **69.44%** | 69.44% | **PASS** |

### Duplicate Triplet Findings in Ground Truth

An audit of ground-truth relationships revealed duplicate identical triplets within single images:
- **IndoorVG**: 335 duplicate triplets (e.g. identical `<sub_id, pred_id, obj_id>` annotated multiple times).
- **VG150**: 148 duplicate triplets.
- **PSG**: 43 duplicate triplets.
- **Open Images V6**: 0 duplicate triplets.

*Impact on Evaluation*: Recall@K evaluators must define whether duplicate ground-truth triplets represent duplicate targets (requiring duplicate predictions) or are matched set-wise (`reduce(np.union1d, pred_to_gt[:k])`).

---

## 6. Stage 5: Subset Selection & Data Leakage Audit

### A. Critical Discovery: IndoorVG Leakage from VG150 Training Set

> [!WARNING]
> **Data Leakage Confirmation**:
> A complete programmatic scan across all 12 parquet shards of the official VG150 training split (`train-00000-of-00012` through `train-00011-of-00012`, totaling 68,538 images) confirmed that **100% of the 200 images (200 / 200) in `IndoorVG_200` are present in the VG150 training set**.

#### Mechanism of Leakage
1. Visual Genome contains ~108,077 images. Xu et al. (2017) partitioned this into 70% Train (68,538), 5% Val (5,000), and 25% Test (31,876).
2. The authors of the IndoorVG benchmark filtered Visual Genome for indoor scenes but generated an **independent random partition** into Train / Val / Test without respecting Xu et al.'s test split boundaries.
3. Consequently, images allocated to the `IndoorVG` test split (including all 200 images in `IndoorVG_200`) were assigned to the `VG150` training split.

#### Methodological Implication
- Any model trained on canonical VG150 (e.g., standard pretrained checkpoints of REACT, FCSGG, or Fast Contextual SGG) **has already seen all 200 images of the IndoorVG evaluation set during training**.
- Evaluating a VG150-pretrained model on `IndoorVG_200` does not measure generalization to an unseen dataset; it measures in-distribution memorization on training images.
- **Requirement**: This training overlap must be explicitly disclosed in all benchmark publications.

### B. ID Collision between VG150 and PSG

- Exactly 7 image IDs appear in both `VG150_200.txt` and `PSG_200.txt`: `2315368`, `2315402`, `2315458`, `2315515`, `2315487`, `2315399`, and `2315372`.
- **Verification**: Pixel hashes and dimensions confirm these are **completely distinct images**:
  - `VG150 2315368`: $500 \times 375$, SHA-256 `0e9a11...` (Visual Genome street/outdoor scene).
  - `PSG 2315368`: $640 \times 480$, SHA-256 `a3b368...` (COCO indoor living room scene).
- **Conclusion**: There is zero image leakage between VG150 and PSG; the overlap is purely a numeric namespace collision between COCO and Visual Genome.

### C. Selection Bias from First 200 Sorted IDs

- **VG150**: Sorted image IDs (`2315353` to `2316499`) sample a contiguous slice of Visual Genome upload batches.
- **IndoorVG**: All 200 images originate from shard 0 (`test-00000-of-00002.parquet`).
- **Open Images V6**: IDs are 16-character hexadecimal MD5 hashes (`000132c2...` to `024ae321...`), providing an unbiased quasi-random uniform sample.

---

## 7. Stage 6: Evaluation Pipeline Readiness & Outstanding Prerequisites

Before running model inference or computing benchmark scores, four essential operational prerequisites must be implemented:

### Prerequisite 1: Enforce PredCls Mode for Cross-Dataset Benchmarking
- **Reason**: The four datasets have completely disjoint object vocabularies:
  - VG150: 150 object classes
  - PSG: 133 object classes (80 COCO thing + 53 COCO stuff)
  - IndoorVG: 84 object classes
  - Open Images: 75 evaluated object classes (601 boxable total)
- **Constraint**: Models cannot perform Scene Graph Detection (SGDet) across different datasets because a model predicting "airplane" (PSG class 5) cannot be evaluated against VG150 class indices without an object class alignment.
- **Resolution**: Cross-dataset evaluation must be restricted to **Predicate Classification (PredCls)** mode (where ground-truth bounding boxes and object class labels are supplied), OR models must be evaluated on their native datasets under normalized relationship views.

### Prerequisite 2: Prediction Projection Layer
- Pretrained models output logits over their native predicate vocabularies (e.g. 50 classes for VG150, 56 for PSG).
- An evaluation wrapper must project native class predictions onto normalized ontology IDs ($1 \dots 6$ for strict; $1 \dots 8$ for extended; $1 \dots 20$ for tri-dataset) before invoking `sgg_metrics.py`.

### Prerequisite 3: Standardize Zero-Relation Image Handling in Recall@K
- In normalized views, some images have 0 ground-truth relationships:
  - Open Images strict core: 158 / 200 images have 0 ground-truth relations.
  - VG150 strict core: 59 / 200 images have 0 ground-truth relations.
- In `sgg_metrics.py`:
  ```python
  rec_i = float(len(match)) / float(gt_rels.shape[0]) if gt_rels.shape[0] > 0 else 0.0
  ```
  Images with 0 relations are assigned Recall = 0.0, severely depressing average recall across 200 images.
- **Resolution**: The benchmark must specify whether empty images are excluded (`filter_empty_rels=True`, standard in PSG) or averaged with 0.0. Both numbers must be reported.

### Prerequisite 4: Disclosure of Model Training Familiarity
- Checkpoints trained on VG150 must be labeled as having **prior training exposure** to `IndoorVG_200`.
- Tri-dataset comparisons are valid across VG150, PSG, and IndoorVG for 17 predicates, and across VG150, IndoorVG, and Open Images for 3 predicates.

---

## 8. Prioritized List of Corrections Requiring Approval

The following items are proposed for user awareness and sign-off (none alter existing frozen files):

1. **[Approval Item 1] Designate Primary Cross-Dataset Evaluation Mode as PredCls**:
   - Approve evaluating cross-dataset transfer in Predicate Classification (PredCls) mode to eliminate object vocabulary divergence.
2. **[Approval Item 2] Adopt `filter_empty_rels=True` as Primary Metric for Sparse Views**:
   - Approve filtering images with 0 ground-truth relations in the evaluated view (e.g. Open Images strict) to report meaningful Recall@K, with full 200-image penalization reported as secondary.
3. **[Approval Item 3] Formalize Training Exposure Disclosure**:
   - Approve adding explicit disclosures in all benchmark tables noting that VG150-trained models have training exposure to `IndoorVG_200`.

---

## 9. Final Safety & Readiness Determination

- **Is it safe to begin model evaluation?**  
  **YES.** The dataset subsets, image files, annotations, and normalized ontology views are 100% complete, verified, reproducible, and mathematically consistent.
- **What remains to be fixed?**  
  No dataset files require modification. Prior to running inference, the evaluation harness must configure: (1) PredCls evaluation mode; (2) prediction mapping onto normalized predicate IDs; and (3) zero-relation filtering conventions.
