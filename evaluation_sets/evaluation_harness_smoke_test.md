# SGG Evaluation Harness Implementation & Smoke-Test Report

**Date**: `2026-10-09`  
**Repository**: `/Users/shivanshkaushik/Developer/SGG-Benchmark`  
**Status**: **IMPLEMENTED & VERIFIED (SMOKE-TEST PASS)**  
**Benchmark Execution Status**: **STANDBY (Full 200-image benchmark intentionally NOT started)**  

---

## 1. Executive Summary

In accordance with the specifications resolved in [`evaluation_sets/evaluation_readiness_report.md`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/evaluation_sets/evaluation_readiness_report.md) and [`evaluation_sets/evaluation_readiness_checklist.json`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/evaluation_sets/evaluation_readiness_checklist.json), the SGG evaluation harness has been implemented, unit-tested on synthetic edge cases, and end-to-end smoke-tested on real evaluation images using the local PSG ONNX model (`react_pp_yolo12m.onnx`).

### Key Accomplishments
1. **Dynamic Ontology Projection Layer**: Created [`sgg_benchmark/data/datasets/evaluation/ontology_projection.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/sgg_benchmark/data/datasets/evaluation/ontology_projection.py). Supports mapping raw native predictions to `strict` (6), `extended` (8), and `tri_dataset` (20) normalized views while strictly preserving subject and object endpoint directionality and filtering unmapped relations.
2. **Dual Recall Ground-Truth Policy**: Updated [`sgg_benchmark/data/datasets/evaluation/sgg_metrics.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/sgg_benchmark/data/datasets/evaluation/sgg_metrics.py) and [`sgg_benchmark/data/datasets/evaluation/sgg_eval.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/sgg_benchmark/data/datasets/evaluation/sgg_eval.py). The primary metric reports Recall@K over eligible images containing $\ge 1$ relation in the target view, and secondary reporting records the all-images penalized recall.
3. **Dynamic Evaluation Vocabulary Sizing**: Replaced hardcoded model-head class counts in `sgg_eval.py` so that `SGMeanRecall` macro-averages over the active evaluation view (e.g., 6 classes in strict core) rather than native model classes.
4. **Architectural & Methodological Safeguards**: Built explicit validation into the evaluation runner:
   - Attempting `PredCls` or `SGCls` on ONNX models raises an immediate `ValueError` explaining that the monolithic graph cannot accept ground-truth boxes or labels.
   - Cross-dataset SGDet without object ontology alignment raises an immediate `ValueError`.
   - VG150 model evaluation on IndoorVG automatically tags the results with `PRIOR_TRAINING_EXPOSURE`.
5. **Comprehensive Verification**:
   - Synthetic unit test suite ([`scratch/test_evaluation_harness.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/scratch/test_evaluation_harness.py)) passed all 9 test cases in 0.015s.
   - End-to-end smoke test on 3 images of `PSG_200` using `react_pp_yolo12m.onnx` executed cleanly in strict evaluation mode.

---

## 2. Code Changes Summary

| File | Change Description | Purpose |
| :--- | :--- | :--- |
| [`sgg_benchmark/data/datasets/evaluation/ontology_projection.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/sgg_benchmark/data/datasets/evaluation/ontology_projection.py) | **New module**: `OntologyProjector` class loading [`evaluation_sets/evaluation_ontology.json`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/evaluation_sets/evaluation_ontology.json). | Implements 2-stage projection (native ID $\rightarrow$ predicate name $\rightarrow$ target view ID $1 \dots K$), filters unmapped predictions, asserts directionality, and enforces model/task compatibility guards. |
| [`sgg_benchmark/data/datasets/evaluation/sgg_metrics.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/sgg_benchmark/data/datasets/evaluation/sgg_metrics.py) | Modified `SGRecall` and `SGMeanRecall`. | Implements dual recall reporting (`_recall` for eligible-only primary, `_recall_all` for all-images secondary) and macro-averages present classes in smaller subsets while also tracking all-classes average. |
| [`sgg_benchmark/data/datasets/evaluation/sgg_eval.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/sgg_benchmark/data/datasets/evaluation/sgg_eval.py) | Updated `num_rel_category` and empty-image evaluator handling. | Binds `num_rel_category` to `len(dataset.ind_to_predicates)` when evaluating normalized views; dispatches empty local containers to `SGRecall` for images with 0 relations. |
| [`sgg_benchmark/data/datasets/data.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/sgg_benchmark/data/datasets/data.py) | Line 167: `rel.get('normalized_predicate_id', rel.get('predicate_id'))`. | Allows `RelationDataset` to load normalized `.coco.json` files seamlessly where predicate IDs are stored under `normalized_predicate_id`. |
| [`tools/evaluate.py`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/tools/evaluate.py) | Integrated CLI flags (`--task`, `--view`, `--config`, `--eval-file`, `--eval-img-dir`, `--max-eval-images`), `OntologyProjector`, peak RSS memory tracking. | Enables parameterized evaluation across tasks (PredCls/SGCls/SGDet) and views (strict/extended/tri/native) with memory and latency instrumentation. |

---

## 3. Unit Test Verification (`scratch/test_evaluation_harness.py`)

A standalone synthetic test suite was executed covering all core mechanisms:

```
Ran 9 tests in 0.015s
OK
```

### Verified Test Cases
1. **`test_vocabulary_sizes`**: Confirmed strict view has 6 predicates, extended has 8, tri-dataset has 20.
2. **`test_psg_projection`**: Confirmed PSG index 4 (`on`) maps to strict ID 5 (`on`), unmapped index 1 (`over`) is filtered out, and relation tensor is resized to $1 \times 7$.
3. **`test_openimages_projection_and_directionality`**: Confirmed Open Images `inside_of` maps to `in` (ID 4) without swapping subject (3) and object (7) endpoints. Confirmed `contain` maps to `has` (tri-dataset ID 7).
4. **`test_dual_reporting_eligible_vs_all`**: On a synthetic 2-image batch (1 image with perfect recall 1.0, 1 image with 0 relations), confirmed primary eligible recall is `1.0` (100%) and secondary all-images recall is `0.5` (50%).
5. **`test_duplicate_ground_truth_matching`**: Confirmed that when ground truth contains identical duplicate triplets, matching logic correctly handles set-union coverage via `np.union1d`.
6. **`test_onnx_predcls_rejected`**: Confirmed requesting `task='predcls'` with an ONNX model raises `ValueError: Task 'predcls' is UNSUPPORTED for ONNX models!`.
7. **`test_onnx_sgcls_rejected`**: Confirmed requesting `task='sgcls'` with an ONNX model raises `ValueError: Task 'sgcls' is UNSUPPORTED for ONNX models!`.
8. **`test_cross_dataset_sgdet_rejected`**: Confirmed requesting SGDet across mismatched datasets (e.g. PSG model on VG150) raises `ValueError: Cross-dataset SGDet is METHODOLOGICALLY INVALID!`.
9. **`test_prior_training_exposure_flagged`**: Confirmed evaluating VG150 model on IndoorVG flags the result as `PRIOR_TRAINING_EXPOSURE`.

---

## 4. End-to-End Smoke Test Execution

### Setup
- **Model Checkpoint**: `checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx`
- **Execution Provider**: `CPUExecutionProvider`
- **Configuration**: `configs/hydra/PSG/REACT++.yaml`
- **Evaluation Dataset**: `evaluation_sets/normalized/PSG_200_strict.coco.json`
- **Evaluation Images**: `evaluation_sets/images/PSG/`
- **Scope**: First 3 images (`max_eval_images = 3`)
- **Evaluation Mode**: `task = sgdet`, `view = strict`
- **Output Directory**: `evaluation_sets/smoke_test_output/`

### Execution Command
```bash
python tools/evaluate.py \
  --model-dir checkpoints/PSG/react++_yolo12m \
  --onnx checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx \
  --config configs/hydra/PSG/REACT++.yaml \
  --provider CPUExecutionProvider \
  --view strict \
  --task sgdet \
  --max-eval-images 3 \
  --num-images 3 \
  --output-dir evaluation_sets/smoke_test_output
```

### Smoke Test Results

```
====================================================================================================
Detection evaluation mAp=0.7132
====================================================================================================
SGG eval:     R @ 20: 0.4667 (all: 0.4667);     R @ 50: 0.4667 (all: 0.4667);     R @ 100: 0.4667 (all: 0.4667);  for mode=sgdet.
SGG eval:    mR @ 20: 0.2778;    mR @ 50: 0.2778;    mR @ 100: 0.2778;  for mode=sgdet.
SGG eval:     zR @ 20: 0.0000 (all: 0.0000);     zR @ 50: 0.0000 (all: 0.0000);     zR @ 100: 0.0000 (all: 0.0000);  for mode=sgdet.
SGG eval:     F1 @ 20: 0.3483;     F1 @ 50: 0.3483;     F1 @ 100: 0.3483;  for mode=sgdet.
====================================================================================================
```

### Latency and Memory Profile
- **Average Forward Latency**: `153.2 ms` (CPU)
- **Median Forward Latency**: `166.1 ms` (CPU)
- **Average E2E Latency**: `156.5 ms`
- **Throughput**: `6.4 FPS` (CPU)
- **Peak RSS Memory**: `885.6 MB`
- **Output Artifact**: [`evaluation_sets/smoke_test_output/eval_summary.json`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/evaluation_sets/smoke_test_output/eval_summary.json)

---

## 5. Constraint Compliance Verification

- [x] **No full benchmark executed**: Exactly 3 images were evaluated for the smoke test; full 200-image evaluation subsets across the four datasets remain untouched.
- [x] **No ground truth or dataset files modified**: All frozen subsets (`VG150_200.*`, `PSG_200.*`, `IndoorVG_200.*`, `OpenImages_200.*`), original annotations, and image files remain unmodified.
- [x] **Ontology decisions strictly preserved**: [`evaluation_sets/evaluation_ontology.json`](file:///Users/shivanshkaushik/Developer/SGG-Benchmark/evaluation_sets/evaluation_ontology.json) was used as read-only authority.
- [x] **Subject/Object directionality verified**: Ensured no endpoint swapping occurs during projection.
- [x] **Empty images policy honored**: Dual reporting verified in unit tests and active in `sgg_metrics.py`.

---

## 6. Next Steps

With the evaluation harness implemented and smoke-tested:
1. Benchmark runs can now be scheduled across the four models (REACT, REACT++, FCSGG, Fast Contextual SGG) across native SGDet and PredCls evaluation modes.
2. Cross-dataset evaluation can be executed in class-agnostic PredCls mode.
3. Comparative tables can be generated projecting outputs to `strict_core_4way` (6), `extended_core_4way` (8), and `tri_dataset_core` (20) vocabularies.
