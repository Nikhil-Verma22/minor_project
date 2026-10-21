# Evidence-Driven Benchmark Audit Discrepancy & Verification Log

**Project**: Object Retrieval in Dynamic Workspace Environments — Cross-Dataset Scene Graph Generation  
**Workspace**: `d:/Minor`  
**Git Revision**: `1b6b15ac7a3186a600d21cf062af6943d1667710` (Working tree state: Modified/Cleaned)  
**Host Architecture**: Intel(R) Core(TM) Ultra 5 125H (14 physical cores, 18 logical threads), 16 GB Physical RAM (15.61 GB visible to OS)  
**Operating System**: Windows 11 Home 64-bit (Build 10.0.26200)  
**Software Stack**: Python 3.13.7, PyTorch 2.8.0+cpu, ONNX Runtime 1.27.0 (`CPUExecutionProvider`)  
**Audit Date**: October 17, 2026  

---

## 1. Executive Summary of Audit Scope

This document provides a systematic, evidence-driven audit of all 16 model–dataset combinations across four architectures (**REACT**, **REACT++**, **FCSGG**, and **Fast Contextual SGG [C-SGG]**) and four benchmark datasets (**OpenPSG [PSG]**, **Visual Genome [VG150]**, **IndoorVG**, and **OpenImages V6**).

Every metric, count, hash, and status recorded here was derived directly from:
1. Physical evaluation annotation files (`evaluation_sets/*_200_annotations.coco.json`).
2. Serialized raw prediction tensor caches (`.cache/*/*.pt`).
3. Model weight checkpoints (`checkpoints/**/*.*`).
4. Standalone benchmark execution runner scripts (`experiments/**/*.py`).
5. Execution logs (`results/**/*.log`).

No metric was assumed, copied from unverified drafts, or retained without bit-for-bit mathematical recomputation against raw prediction tensors.

---

## 2. Verification Status Taxonomy & Criteria

To ensure complete scientific integrity, unconditional assignments of verification (e.g. `True` or `COMPLETED`) have been permanently removed. Every run is evaluated under two distinct, criteria-driven statuses:

| Status Name | Exact Operational Criteria | Audit Application |
| :--- | :--- | :--- |
| **`VERIFIED_FROM_RAW_CACHE`** | (1) Expected serialized prediction cache exists on disk and is nonzero in size.<br>(2) Cache contains predictions for all 200 evaluation images.<br>(3) Target dataset annotation file exists, parses valid COCO JSON structure, and matches verified SHA-256.<br>(4) Independent recomputation of bounding box IoU matching (threshold 0.50), triplet matching, and recall aggregation reproduces reported JSON metrics within $|\Delta| < 10^{-4}$. | **Applied to all 16 runs**: Every run satisfies all 4 conditions with a maximum absolute error $|\Delta| \le 5.0 \times 10^{-5}$ across all recall and mean recall metrics. |
| **`REPRODUCED_END_TO_END`** | Fresh, live model forward inference was executed from scratch on the CPU across all 200 evaluation images using the documented checkpoint, matching predictions and timing within latency tolerances. | **Not Assigned in this Audit Pass**: Fresh end-to-end CPU re-inference across 3,200 forward passes (~6 hours of compute) was not re-executed during this audit session. |
| **`NOT_EXECUTED_IN_AUDIT`** | End-to-end forward pass was previously executed in historical benchmark runs (evidenced by logs), but live re-inference was deliberately omitted in the current audit to preserve compute. | **Applied as End-to-End Status** across all 16 runs. |
| **`PARTIALLY_VERIFIED`** | Only a subset of metrics or files could be verified, with specific remaining gaps documented. | None. All 16 runs passed full cache-level verification. |
| **`NOT_INDEPENDENTLY_VERIFIABLE_FROM_AVAILABLE_ARTIFACTS`** | Checkpoints, configurations, or raw tensors are partially missing, preventing independent mathematical recomputation. | Reserved for incomplete runs. (All 16 core runs had complete artifacts available). |
| **`UNVERIFIED`** | Required checkpoints, prediction caches, or annotations are completely absent or unreadable. | None. All required files exist and parse successfully. |
| **`FAILED`** | Recomputation executed but produced metric divergence exceeding numerical tolerance ($|\Delta| \ge 10^{-4}$). | None. Zero runs failed. |

---

## 3. Discrepancy Log: Historical Conflicts, Root Causes, and Decisions

### Discrepancy 1: IndoorVG Annotation Checksum Listed as `e3b0c442...` (Empty Byte Sequence)
- **Previous Value**: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- **Current Verified Value**: `6f377bca5f7409b346333e8997047b95b62e7f82e2760a905ceaef7693d348a1`
- **Raw-Source Evidence**: Direct inspection of [`evaluation_sets/IndoorVG_200_annotations.coco.json`](file:///d:/Minor/evaluation_sets/IndoorVG_200_annotations.coco.json) on disk revealed a valid, nonempty file of **965,773 bytes** containing 200 images, 84 object categories, 3,118 bounding box annotations, and 1,672 relationship annotations (`rel_annotations`).
- **Exact Cause**: `e3b0c442...` is the exact mathematical SHA-256 digest of an empty byte string (`hashlib.sha256(b"").hexdigest()`). A legacy generator script had an uninitialized path variable for IndoorVG when computing hashes, producing the empty-string hash.
- **Final Decision & Action**: Recomputed the SHA-256 digest directly from the file bytes on disk. Replaced all occurrences in audit records, matrix JSON, and documentation with `6f377bca5f7409b346333e8997047b95b62e7f82e2760a905ceaef7693d348a1`.
- **Remaining Uncertainty**: None.

---

### Discrepancy 2: REACT on OpenImages V6: R@100 and mR@100 Metric Values & Aggregation Formulas
- **Previous Conflicting Values**:
  - Legacy markdown draft: `R@100 = 1.78%`, `mR@100 = 10.76%` (identical to R@50 / mR@50).
  - Metric file [`results/REACT/openimages_metrics.json`](file:///d:/Minor/results/REACT/openimages_metrics.json): `R@100 = 1.90%`, `mR@100 = 12.05%`.
- **Raw-Source Evidence & Arithmetic Derivation**: Independent recomputation from [`.cache/REACT/raw_preds_OpenImages_react_full.pt`](file:///d:/Minor/.cache/REACT/raw_preds_OpenImages_react_full.pt) (4,147,703 bytes, SHA-256 `1dca6988284b738c...`) across all 200 evaluation images (866 total ground-truth relations) establishes:
  1. **Matched True-Positive Hits**:
     - At Top 50 ($K=50$): Exactly **18 unique matched ground-truth triplets**.
     - At Top 100 ($K=100$): Exactly **20 unique matched ground-truth triplets** (+2 new true positives in ranks 51..100).
  2. **Conventional Pooled Micro-Recall** ($\frac{\sum \text{Hits}}{\sum \text{GT}}$):
     - At $K=50$: $\frac{18}{866} = \mathbf{2.08\%}$ ($0.020785$).
     - At $K=100$: $\frac{20}{866} = \mathbf{2.31\%}$ ($0.023095$).
  3. **Image-Averaged Recall** ($\frac{1}{|\mathcal{I}_{\text{eligible}}|}\sum_{i=1}^{|\mathcal{I}|} \frac{\text{Hits}_i}{\text{GT}_i}$ across 200 eligible images):
     - At $K=50$: Sum of per-image recalls $= 3.5583 \implies \frac{3.5583}{200} = \mathbf{1.78\%}$ ($0.017792$).
     - At $K=100$: Sum of per-image recalls $= 3.7917 \implies \frac{3.7917}{200} = \mathbf{1.90\%}$ ($0.018958$).
  4. **Image-Averaged Macro Mean Recall** ($\text{mR}^{\text{img-avg}}$):
     - At $K=50$: Macro average across 13 active predicate classes $= \mathbf{10.76\%}$ ($0.107639$).
     - At $K=100$: Macro average across 13 active predicate classes $= \mathbf{12.05\%}$ ($0.120460$).
  5. **Conventional Pooled Macro Mean Recall** ($\text{mR}^{\text{pooled}}$):
     - At $K=50$: $\mathbf{8.61\%}$ ($0.086106$).
     - At $K=100$: $\mathbf{9.31\%}$ ($0.093099$).
- **Exact Cause of Discrepancy**:
  - The legacy markdown draft suffered a copy-paste error where R@50 and mR@50 were accidentally duplicated into the R@100 and mR@100 cells.
  - Furthermore, earlier discussions conflated the image-averaged recall (1.78% / 1.90%) with the pooled quotient (18/866 = 2.08% / 20/866 = 2.31%). These are not contradictory measurements, but two distinct statistical aggregations: pooled recall weights every relation equally, whereas image-averaged recall weights every image equally. Images with many relations where the model scored zero pull the image average down relative to the pooled quotient.
- **Final Decision & Action**:
  - Restored verified metrics across all deliverables with explicit aggregation labels:
    - **Image-Averaged R@50: 1.78%**, **Image-Averaged R@100: 1.90%** (200 eligible images).
    - **Pooled Micro-Recall R@50: 2.08%**, **Pooled Micro-Recall R@100: 2.31%** (866 total relations).
    - **Image-Averaged mR@50: 10.76%**, **Image-Averaged mR@100: 12.05%**.
- **Remaining Uncertainty**: None.

---

### Discrepancy 3: REACT++ on VG150: R@100 and mR@100 Discrepancy (26.68% / 18.25% vs 25.56% / 18.48%)
- **Previous Conflicting Values**:
  - Legacy multi-dataset text: `R@100 = 26.68%`, `mR@100 = 18.25%`.
  - Metric file [`results/REACTPP/vg150_metrics.json`](file:///d:/Minor/results/REACTPP/vg150_metrics.json): `R@100 = 25.56%`, `mR@100 = 18.48%`.
- **Raw-Source Evidence**: Evaluated [`.cache/REACTPP/raw_preds_vg150_reactpp.pt`](file:///d:/Minor/.cache/REACTPP/raw_preds_vg150_reactpp.pt) (9,559,487 bytes, SHA-256 `0ff70777ec758878...`) using the script's native `pred_triplet_scores` ranking rule:
  - Top 50 candidates: **288 matched hits** $\to$ $\text{R@50} = 23.54\%$, $\text{mR@50} = 16.57\%$.
  - Top 100 candidates: **310 matched hits** (+22 new true positive matches) $\to$ $\text{R@100} = 25.56\%$, $\text{mR@100} = 18.48\%$.
- **Exact Cause**: The 26.68% figure was an unverified text draft from an exploratory post-processing experiment that was never integrated into the official benchmark script.
- **Final Decision & Action**: Standardized on the reproducible, cache-verified values: **R@50: 23.54%**, **R@100: 25.56%**, **mR@50: 16.57%**, **mR@100: 18.48%**.
- **Remaining Uncertainty**: None.

---

### Discrepancy 4: Suspicion of Cache Reuse Between REACT and REACT++ on IndoorVG
- **Issue**: An earlier progress note flagged that REACT and REACT++ on IndoorVG both had identical metrics (`R@50 = 31.08%`, `R@100 = 36.58%`, `latency = 724.8 ms`), raising suspicion of shared prediction caches.
- **Raw-Source Evidence**:
  - `checkpoints/REACT/react_indoorvg_yolov8m.onnx`: 138,878,957 bytes, SHA-256 `5188b578aa155e35...`
  - `checkpoints/REACT/react_pp_vg150_yolo12m.onnx`: 116,984,567 bytes, SHA-256 `ce01616b902271da...`
  - `.cache/REACT/raw_preds_indoorvg_react.pt`: 7,495,165 bytes, SHA-256 `ea34568c67b8e102...`
  - `.cache/REACTPP/raw_preds_indoorvg_reactpp.pt`: 4,512,297 bytes, SHA-256 `6e755d72b8e05c86...`
  - Comparing image predictions: In image 1, REACT has 100 boxes, REACT++ has 50 boxes; in image 2, REACT has 100 boxes, REACT++ has 81 boxes.
  - Execution log `results/REACTPP/indoorvg_run.log` shows the full run completed on October 17 at 01:18:25, recording **2450.7 ms (0.41 FPS)** and producing:
    - $\text{R@50} = 13.64\%$, $\text{R@100} = 16.02\%$, $\text{mR@50} = 12.69\%$, $\text{mR@100} = 14.07\%$.
- **Exact Cause**: The earlier note observed a placeholder entry in an initial draft table where Row 2 had been temporarily copied into Row 6 before the REACT++ IndoorVG benchmark script had finished executing.
- **Final Decision & Action**: Confirmed that REACT and REACT++ on IndoorVG are completely independent, have distinct prediction caches, distinct checkpoints, and distinct metrics.
- **Remaining Uncertainty**: None.

---

### Discrepancy 5: FCSGG on VG150 (9.08% vs 20.37%)
- **Previous Conflicting Values**: Legacy reports referenced 20.37% R@50 alongside 9.08% R@50.
- **Raw-Source Evidence**:
  - Evaluation of [`.cache/FCSGG/raw_preds_VG150_full.pt`](file:///d:/Minor/.cache/FCSGG/raw_preds_VG150_full.pt) with [`evaluation_sets/VG150_200_annotations.coco.json`](file:///d:/Minor/evaluation_sets/VG150_200_annotations.coco.json) under Scene Graph Detection (SGDet: predicted boxes + predicted classes + predicted relations) yields exactly:
    - $\text{R@50} = 9.08\%$, $\text{R@100} = 12.23\%$, $\text{mR@50} = 2.62\%$, $\text{mR@100} = 3.26\%$.
  - Literature inspection of Liu et al. (CVPR 2021) shows that 20.37% is the **Predicate Classification (PredCls)** score under ground-truth bounding boxes and object classes, not SGDet.
- **Final Decision & Action**: Retained the true empirical SGDet scores (**9.08% / 12.23%**) and explicitly documented that 20.37% was a cited PredCls figure from the original paper.
- **Remaining Uncertainty**: None.

---

### Discrepancy 6: C-SGG on VG150 (20.05% vs 18.78%)
- **Previous Conflicting Values**: Legacy reports cited 18.78% R@50 in one table and 20.05% R@50 in another.
- **Raw-Source Evidence**: Re-evaluating [`.cache/FastContextualSGG/raw_preds_VG150_full.pt`](file:///d:/Minor/.cache/FastContextualSGG/raw_preds_VG150_full.pt) with different candidate sorting rules showed:
  - If pairs are sorted by predicate score alone ($s_{\text{rel}}$): $\text{R@50} = 18.78\%$, $\text{mR@50} = 13.91\%$.
  - If pairs are sorted by unified triplet confidence ($S = s_{\text{subj}} \times s_{\text{obj}} \times s_{\text{rel}}$ as implemented in `experiments/FastContextualSGG/run_benchmark_fast_contextual_vg150.py`): $\text{R@50} = 20.05\%$, $\text{R@100} = 24.85\%$, $\text{mR@50} = 14.10\%$, $\text{mR@100} = 15.33\%$.
- **Final Decision & Action**: Standardized on the official joint-confidence metric (**20.05% / 24.85%**) implemented in the benchmark runner script.
- **Remaining Uncertainty**: None.

---

### Discrepancy 7: Detector Architecture Contradictions for Fast Contextual SGG (C-SGG)
- **Previous Contradiction**: Different report versions described C-SGG as using "Faster R-CNN ResNet-50-FPN" or "YOLOv5l + CKN".
- **Raw-Source Evidence**:
  - Checkpoint file [`checkpoints/FastContextualSGG/yolol_object_test_28.pt`](file:///d:/Minor/checkpoints/FastContextualSGG/yolol_object_test_28.pt) (376,488,983 bytes, SHA-256 `f657e21917f10d2d...`) was inspected with PyTorch.
  - Model class is `models.yolo.Model` (YOLOv5l architecture, `nc=151`, corresponding to Visual Genome 150 object classes plus background).
  - Code in `experiments/FastContextualSGG/fast_contextual_adapter.py` lines 53-59 explicitly instantiates this model and loads `yolol_object_test_28.pt`.
- **Exact Cause**: Faster R-CNN was the generic baseline cited from literature in the background section; the actual detector implemented and executed in this repository is **YOLOv5l**.
- **Final Decision & Action**: Replaced all mentions of Faster R-CNN for C-SGG with **YOLOv5l** across all tables and reports.
- **Remaining Uncertainty**: None.

---

### Discrepancy 8: Evaluation Regime Provenance Contradictions
- **Previous Contradiction**: FCSGG on PSG was previously described as "in-domain" with a nonexistent checkpoint `fcsgg_psg.pth`, while REACT++ IndoorVG was labeled without clear provenance.
- **Raw-Source Evidence**:
  - Checkpoint inspection confirmed no `fcsgg_psg.pth` exists in the repository.
  - `experiments/FCSGG/run_benchmark_fcsgg_psg.py` loads `checkpoints/FCSGG/HRNetW32-1S.pth` (VG150-trained) and constructs an explicit cross-dataset semantic ontology mapping from VG150 to OpenPSG.
  - `experiments/FastContextualSGG/run_benchmark_fast_contextual_psg.py` loads the YOLO12m detector from `checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx` combined with the OpenPSG CKN model `model_repos/fast-context-scene-graph-generation/PSG_dataset/ckn/2619.pt`.
  - `experiments/FastContextualSGG/run_benchmark_fast_contextual_openimages.py` loads the VG150-trained YOLOv5l detector combined with the native OpenImages CKN model `model_repos/fast-context-scene-graph-generation/OID_dataset/ckn/ckn_oid_PBS.pt`.
- **Final Decision & Action**: Rigorously classified all 16 runs into three provenanced regimes:
  1. **In-Domain**: REACT (PSG, VG150, IndoorVG Native ONNX), REACT++ (PSG, VG150), FCSGG (VG150), C-SGG (VG150).
  2. **Zero-Shot Transfer**: REACT (OpenImages), REACT++ (IndoorVG, OpenImages), FCSGG (PSG, IndoorVG, OpenImages), C-SGG (IndoorVG).
  3. **Hybrid Adaptation**: C-SGG on PSG (YOLO12m detector + OpenPSG CKN), C-SGG on OpenImages (VG150 YOLOv5l detector + OID CKN).
- **Remaining Uncertainty**: None.

---

### Discrepancy 9: Host RAM Overstatement (32 GB vs 16 GB)
- **Previous Claim**: Methodology and reports claimed the benchmark ran on an Intel Core Ultra 5 125H with "32 GB RAM".
- **Raw-Source Evidence**: Live machine inspection via Windows CIM (`Get-CimInstance Win32_OperatingSystem`) and Python `psutil.virtual_memory().total` revealed:
  - Total physical memory: 16,761,901,056 bytes (**15.61 GB visible to OS**).
  - CPU: Intel(R) Core(TM) Ultra 5 125H (14 cores: 4 Performance, 8 Efficient, 2 Low-Power Efficient; 18 logical processors).
- **Exact Cause**: Typographical overstatement in early markdown documentation.
- **Final Decision & Action**: Corrected host specification across all files to **16 GB RAM (15.61 GB visible to OS)**.
- **Remaining Uncertainty**: None.

---

### Discrepancy 10: Misleading Triplet F1 and Detection F1 Terminology
- **Previous Claim**: Tables presented $2 \cdot \frac{R@50 \cdot mR@50}{R@50 + mR@50}$ as "F1@50" and detection harmonic mean as "Detection F1".
- **Exact Cause**: Mean recall ($mR$) is macro-averaged recall across classes, not precision. Calling the harmonic mean of micro-recall and macro-recall an "F1 score" violates standard statistical definitions.
- **Final Decision & Action**:
  - Relabeled SGG recall balance to **`Supplementary Harmonic Recall H(R,mR)@50`** with disclosed formula $2 \cdot \frac{R \cdot mR}{R + mR}$.
  - Relabeled detection metric to **`Detection H(AP,AR)@50`**, representing the harmonic mean of COCO detection mAP@50 and mAR@50 (at maxDets=100, IoU=0.50).
- **Remaining Uncertainty**: None.

---

## 4. Mathematical Proof for Equal R@50 and R@100 Runs

Two runs in the benchmark produce identical R@50 and R@100 values:
1. **FCSGG on OpenImages V6 (Row 11)**: $\text{R@50} = 0.80\%$, $\text{R@100} = 0.80\%$.
2. **Fast Contextual SGG on OpenImages V6 (Row 15)**: $\text{R@50} = 2.45\%$, $\text{R@100} = 2.45\%$.

### Empirical Verification:
To verify whether this equality reflects a genuine property of the predictions or a truncation bug, we conducted an exhaustive count of unique matched ground-truth triplets across candidate rank thresholds:

```
Run: FCSGG on OpenImages V6 (200 images, 866 GT relations)
- Matched True Positive Hits at Top 50 (K=50):   9 hits
- Matched True Positive Hits at Top 100 (K=100):  9 hits
- New Matched Hits captured in Ranks 51..100:     0 hits (+0.0%)
- Calculation: R@50 = 9 / 866 = 0.008083 (0.80%)
               R@100 = 9 / 866 = 0.008083 (0.80%)

Run: Fast Contextual SGG on OpenImages V6 (200 images, 866 GT relations)
- Matched True Positive Hits at Top 50 (K=50):   18 hits
- Matched True Positive Hits at Top 100 (K=100):  18 hits
- New Matched Hits captured in Ranks 51..100:     0 hits (+0.0%)
- Calculation (Image-Averaged): R@50 = 0.024458 (2.45%)
                                R@100 = 0.024458 (2.45%)
- Calculation (Pooled):         R@50 = 18 / 866 = 0.020785 (2.08%)
                                R@100 = 18 / 866 = 0.020785 (2.08%)
```

**Root Cause**: Under zero-shot transfer onto OpenImages V6, the object detectors propose a sparse set of valid overlapping bounding boxes that match OpenImages object semantics. High-scoring true positive box pairings are concentrated entirely within the top 50 ranked candidates. Candidate pairs ranked 51 to 100 consist exclusively of low-confidence or false-positive box pairings, producing exactly zero incremental hits between ranks 51 and 100.

---

## 5. OpenImages V6 Dual-Protocol Predicate Audit

### 5.1 Verification of Predicate `is` (Category ID 27)
- Inspection of `evaluation_sets/OpenImages_200_annotations.coco.json` and `dataset.ind_to_predicates` confirms that index **27** corresponds to predicate **`is`**.
- In OpenImages V6, `is` represents **intra-object visual attributes** (e.g. `[Woman, is, Smiling]`, `[Car, is, White]`, `[Table, is, Wooden]`).
- Visual attribute annotations have identical subject and object bounding box coordinates ($s = o$) or annotate an internal attribute token.
- Standard scene graph generation architectures (REACT, FCSGG, C-SGG) predict directed relationships between two distinct physical objects ($s \neq o$). They possess no attribute prediction head.
- As a mathematical consequence, **all models score 0 hits across all 720 `is` annotations**.

### 5.2 Side-by-Side Dual-Protocol Evaluation

| Evaluation Metric | Protocol A: Original Population (Literature Standard) | Protocol B: Attribute-Excluded Population (Physical Relations Only) | Mathematical Relationship & Derivation |
| :--- | :---: | :---: | :--- |
| **Annotation Filtering Rule** | None (All 866 ground-truth relations retained) | Relations with `predicate_id == 27` (`is`) removed | 720 unary attribute relations excluded |
| **Total Ground-Truth Relations** | **866 relations** | **146 relations** | $866 - 720 = 146$ physical relations |
| **Eligible Image Population** | **200 / 200 images** (100.0%) | **58 / 200 images** (29.0%) | 142 images contain strictly visual attributes |
| **Matched Hits at Top 50** | **18 hits** | **18 hits** | All 18 true positive hits are physical relations |
| **Matched Hits at Top 100** | **18 hits** (+0 new hits) | **18 hits** (+0 new hits) | No new hits in ranks 51..100 |
| **Image-Averaged Recall R@50** | **2.45%** ($0.024458$) | **15.47%** ($0.154652$) | $\frac{1}{200}\sum \frac{\text{Hits}_i}{\text{GT}_i}$ vs $\frac{1}{58}\sum \frac{\text{Hits}_i}{\text{GT}_i^{\text{phys}}}$ |
| **Image-Averaged Recall R@100** | **2.45%** ($0.024458$) | **15.47%** ($0.154652$) | Zero incremental recall at top 100 |
| **Conventional Pooled Micro-Recall R@50** | **2.08%** ($\frac{18}{866} = 0.020785$) | **12.33%** ($\frac{18}{146} = 0.123288$) | $\frac{\sum \text{Hits}}{\sum \text{GT}}$ directly reflects the reduced denominator |
| **Image-Averaged Macro Mean Recall mR@50** | **24.03%** ($0.240349$) | **26.04%** ($0.260379$) | Macro average across active classes (13 vs 12 classes) |
| **Conventional Pooled Macro Mean Recall mR@50** | **22.39%** ($0.223858$) | **24.25%** ($0.242513$) | Macro average of pooled class recalls |

### 5.3 Complete Per-Predicate Breakdown (C-SGG on OpenImages V6)

The table below audits every active predicate in OpenImages V6 and explains why **Image-Averaged Recall** differs from **Pooled Micro-Recall**:

| Predicate Name | Predicate Index | Ground-Truth Count ($\sum \text{GT}$) | Active Images ($|\mathcal{I}_c|$) | Matched Hits@50 | Conventional Pooled Recall ($\frac{\text{Hits}}{\sum \text{GT}}$) | Image-Averaged Recall ($\frac{1}{|\mathcal{I}_c|}\sum \frac{\text{Hits}_i}{\text{GT}_i}$) | Mathematical Explanation of Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `at` | 1 | 22 | 8 | 8 | **36.36%** (8 / 22) | **38.10%** | Hits distributed across 8 images; per-image recall average is 38.10%. |
| `holds` | 2 | 18 | 13 | 1 | **5.56%** (1 / 18) | **7.69%** | 1 hit in 1 image out of 13 active images ($1/13 \approx 7.69\%$). |
| `wears` | 3 | 47 | 20 | 2 | **4.26%** (2 / 47) | **7.50%** | 2 hits across 20 active images; per-image average is 7.50%. |
| `hang` | 5 | 3 | 2 | 0 | **0.00%** (0 / 3) | **0.00%** | Zero true positive matches. |
| `on` | 8 | 12 | 8 | 0 | **0.00%** (0 / 12) | **0.00%** | Zero true positive matches. |
| `ride` | 9 | 14 | 5 | 2 | **14.29%** (2 / 14) | **13.33%** | 2 hits across 5 active images; per-image average is 13.33%. |
| `skateboard` | 11 | 1 | 1 | 1 | **100.00%** (1 / 1) | **100.00%** | 1 hit on 1 single GT relation in 1 image. |
| `inside_of` | 14 | 4 | 3 | 1 | **25.00%** (1 / 4) | **33.33%** | 1 hit in 1 image out of 3 active images ($1/3 \approx 33.33\%$). Earlier table incorrectly claimed $1/4 = 33.33\%$. |
| `contain` | 17 | 18 | 8 | 1 | **5.56%** (1 / 18) | **12.50%** | 1 hit in 1 image out of 8 active images ($1/8 = 12.50\%$). Denominators differ. |
| `interacts_with` | 21 | 2 | 2 | 2 | **100.00%** (2 / 2) | **100.00%** | 2 hits across 2 active images. |
| `is` *(attribute)* | 27 | 720 | 185 | 0 | **0.00%** (0 / 720) | **0.00%** | Bounding box pair models have no attribute head. |
| `kick` | 28 | 1 | 1 | 0 | **0.00%** (0 / 1) | **0.00%** | Zero true positive matches. |
| `plays` | 30 | 4 | 3 | 0 | **0.00%** (0 / 4) | **0.00%** | Zero true positive matches. |
| **Total (Original)** | — | **866** | **200** | **18** | **2.08%** (18 / 866) | **24.03%** *(macro mR)* | $\text{mR}^{\text{img-avg}} = \frac{3.12454}{13} = \mathbf{24.03\%}$; $\text{mR}^{\text{pooled}} = \mathbf{22.39\%}$. |
| **Total (Excl. `is`)**| — | **146** | **58** | **18** | **12.33%** (18 / 146)| **26.04%** *(macro mR)* | $\text{mR}^{\text{img-avg}} = \frac{3.12454}{12} = \mathbf{26.04\%}$; $\text{mR}^{\text{pooled}} = \mathbf{24.25\%}$. |

---

## 6. Cryptographic Artifact Inventory

### 6.1 Evaluation Annotations
| Dataset | Relative Path | Byte Size | SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| **PSG** | `evaluation_sets/PSG_200_annotations.coco.json` | 919,338 | `624521593199008f36499cda8e2f12d615c30714a94865d23c5654f92d712327` |
| **VG150** | `evaluation_sets/VG150_200_annotations.coco.json` | 707,327 | `8a969a879eb4d9982eafa4d2e02ecfe367f1c0737d56c6ce380bc8cff10efc5c` |
| **IndoorVG** | `evaluation_sets/IndoorVG_200_annotations.coco.json` | 965,773 | `6f377bca5f7409b346333e8997047b95b62e7f82e2760a905ceaef7693d348a1` |
| **OpenImages** | `evaluation_sets/OpenImages_200_annotations.coco.json` | 560,765 | `20eed09f1e2bd454f9324a28e2a6871e6e8d21f73352302addf09b1c27590b33` |

### 6.2 Checkpoints and Component Weights
| Component Description | Relative Path | Byte Size | SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| **REACT PSG Best Model** | `checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth` | 235,926,680 | `0f27511e66f77f6d3eac4c09f5ccb06a61e702547032a2eddb1b601fe26ff03d` |
| **REACT VG150 Best Model** | `checkpoints/REACT/react_VG150/best_model_epoch_9.pth` | 235,981,109 | `8bf598afdb2ccef9e5b6f54dc1fe0fa70b46a4bfceecfb02f693f2e82ad503bf` |
| **REACT IndoorVG Native ONNX** | `checkpoints/REACT/react_indoorvg_yolov8m.onnx` | 138,878,957 | `5188b578aa155e3584d95b0e3d87a5bbd80ae2464f6d1f3e380d9b7ad872c7e8` |
| **REACT++ PSG ONNX** | `checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx` | 116,791,801 | `38c4bdbb7e2a077131ffcbf57af726eb30574318e2344d5b4bfc1d3a48f5b2fe` |
| **REACT++ VG150 ONNX** | `checkpoints/REACT/react_pp_vg150_yolo12m.onnx` | 116,984,567 | `ce01616b902271da224c61ff85f9914f8efb12c8894e4aa7bad42ebe9480c9f2` |
| **FCSGG HRNetW32-1S Checkpoint**| `checkpoints/FCSGG/HRNetW32-1S.pth` | 380,012,898 | `41aa7639bac0793ef6ad04b860b2d441a32523cf0f69b7590b2eda261638c084` |
| **C-SGG YOLOv5l Detector** | `checkpoints/FastContextualSGG/yolol_object_test_28.pt` | 376,488,983 | `f657e21917f10d2d70da5e3f1b02d3148dc8589ac74ba6e7d8c2fc5e2aaac257` |
| **C-SGG OpenPSG CKN Weights** | `model_repos/fast-context-scene-graph-generation/PSG_dataset/ckn/2619.pt` | 840,791 | `3a93b2948c7406c7c3f281a212b65d7c39a19aa706883c1b7f8fc3bba71b9efe` |
| **C-SGG VG150 CKN Weights** | `model_repos/fast-context-scene-graph-generation/VG_dataset/ckn/CKN+EPBS.pt` | 828,503 | `ffddecd41adf7881b215f6dd186c9c7b18d5a59a33b4574d71687200c4e330c4` |
| **C-SGG OpenImages CKN Weights**| `model_repos/fast-context-scene-graph-generation/OID_dataset/ckn/ckn_oid_PBS.pt` | 787,415 | `c106480f718cf6d8e581feff42df43916fc3cbcbbfd5d2ae8df7ff2e30cb6a6a` |
| **PSG Word Vectors** | `.cache/FastContextualSGG/psg_word_vec_list.pt` | 28,399 | `dd42ac400e8e6c41f3cfbac235848f03f456c58494a26ed2fe5824bffc44be22` |
| **VG150 Word Vectors** | `.cache/FastContextualSGG/vg_word_vec_list.pt` | 31,848 | `a3bc639414c53b20d2b48a1eff8efb55a46ab05adbc59137f625c8d085ccd47d` |

### 6.3 Serialized Raw Prediction Caches
| Run ID | Prediction Cache Path | Byte Size | SHA-256 Digest | Cache Uniqueness |
| :--- | :--- | :---: | :--- | :---: |
| **RUN-01** | `.cache/REACT/raw_preds_PSG_full.pt` | 5,684,009 | `8dc8f536306cdde18b625ee965d137351fc8645a0b784a32ec4283838186f916` | Unique |
| **RUN-02** | `.cache/REACT/raw_preds_VG150_full.pt` | 5,440,709 | `e0db14dc08779b119c438185c783db9f9e160a22c54cb4f55cb3b0f5139420dc` | Unique |
| **RUN-03** | `.cache/REACT/raw_preds_indoorvg_react.pt` | 7,495,165 | `ea34568c67b8e1022d3cbefa537980f672851d4b8a999d9bd0180982c6458b0a` | Unique |
| **RUN-04** | `.cache/REACT/raw_preds_OpenImages_react_full.pt` | 4,147,703 | `1dca6988284b738cffbf630e668c2ee03e33c6ce09632a6774659223e7cfad6d` | Unique |
| **RUN-05** | `.cache/REACTPP/raw_preds_PSG_full.pt` | 3,879,465 | `30080c68729f594586dcf35a111a4db95945037d80ae526fbfa3298a8553e198` | Unique |
| **RUN-06** | `.cache/REACTPP/raw_preds_vg150_reactpp.pt` | 9,559,487 | `0ff70777ec758878891d4e0e47087050a41753c155513d6a9bfe4122d4f8fcad` | Unique |
| **RUN-07** | `.cache/REACTPP/raw_preds_indoorvg_reactpp.pt` | 4,512,297 | `6e755d72b8e05c862ce54516ce88412832ca4d02457d1508b524a00d832afe45` | Unique |
| **RUN-08** | `.cache/REACTPP/raw_preds_OpenImages_reactpp_full.pt` | 4,047,523 | `1d83a28fe07caf4a2a16d5668b556cc46237894a46979603e99480a424266205` | Unique |
| **RUN-09** | `.cache/FCSGG/raw_preds_PSG_fcsgg_full.pt` | 10,069,565 | `23dde72fbd10ac68a05c6d32ffeb6d2238ceca9d4c22fb09710ebca873539958` | Unique |
| **RUN-10** | `.cache/FCSGG/raw_preds_VG150_full.pt` | 5,457,577 | `7e676f699256d17676fb4bb152ce8ec84c47285a73e6f9621b18d2da8cb54ec5` | Unique |
| **RUN-11** | `.cache/FCSGG/raw_preds_IndoorVG_fcsgg_full.pt` | 8,553,395 | `4019c0b85cce2c86e0c65651ec3c52e4ecfe086c87e148db2923769970ae513f` | Unique |
| **RUN-12** | `.cache/FCSGG/raw_preds_OpenImages_fcsgg_full.pt` | 3,944,631 | `45a21705f9452e29e92a83e020257c7ee00c0f3a61fba96be7dfbe984bf4244d` | Unique |
| **RUN-13** | `.cache/FastContextualSGG/raw_preds_PSG_csgg_full.pt` | 4,726,791 | `489a4e5fc226f73a5a75b7b629e46a788bb25e6f3d9198642a8b981502444e2b` | Unique |
| **RUN-14** | `.cache/FastContextualSGG/raw_preds_VG150_full.pt` | 5,141,977 | `db1fb1df158a4b59e355f32bdf61ae84334f59ef262e3dffc79f94ebdb79c2e0` | Unique |
| **RUN-15** | `.cache/FastContextualSGG/raw_preds_IndoorVG_c-sgg_full.pt` | 8,109,707 | `77b2f28d3b3fa18e6a1005a8f4c2c0a4e70e5b7fb582ae1230cbfd9d40131758` | Unique |
| **RUN-16** | `.cache/FastContextualSGG/raw_preds_OpenImages_c-sgg_full.pt` | 7,692,023 | `8430e40c86c012e1f40d7c35d96f9c464bf27b99c0daae84d9f7a77e9fb1b550` | Unique |

---

## 7. Final Quality Control Checklist

- [x] All 16 run IDs match between [`results/single_pair_benchmark_matrix.csv`](file:///d:/Minor/results/single_pair_benchmark_matrix.csv) and [`results/reproducibility_audit_records.json`](file:///d:/Minor/results/reproducibility_audit_records.json).
- [x] All 16 cache recomputations match reported JSON metrics with absolute error $|\Delta| < 10^{-4}$ (maximum error across all runs: $5.0 \times 10^{-5}$).
- [x] All four annotation files exist, are nonempty, parse valid COCO structures, and match their documented cryptographic checksums.
- [x] Unconditional verification statuses permanently removed; replaced with `VERIFIED_FROM_RAW_CACHE` and `NOT_EXECUTED_IN_AUDIT`.
- [x] Hardware host specifications reconciled to 16 GB physical RAM (15.61 GB visible to OS).
- [x] OpenImages original (866 relations) and attribute-excluded (146 relations) protocols audited side-by-side.
- [x] Mathematical cause of R@50 == R@100 on OpenImages documented and proven.
