import os
import sys
import time
import json
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from functools import reduce

import torch
import numpy as np
import psutil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

WORKSPACE_ROOT = "d:/Minor"
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/SGG-Benchmark"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fcsgg/detectron2"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fcsgg"))

from sgg_benchmark.data.datasets.data import RelationDataset
from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
)
from experiments.REACT.ontology_projection import OntologyProjector
from experiments.FCSGG.fcsgg_adapter import FCSGGAdapter

# Directories
LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/FCSGG")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/FCSGG")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("FCSGGBenchmark")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(os.path.join(LOG_DIR, "run.log"), mode="a", encoding="utf-8")
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

def get_peak_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)

def run_fcsgg_inference(
    adapter: FCSGGAdapter,
    dataset: RelationDataset,
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Executes FCSGG inference sequentially over the evaluation dataset.
    """
    num_images = min(max_images, len(dataset))
    logger.info(f"Running FCSGG inference across {num_images} images...")

    predictions = []
    fwd_times = []
    e2e_times = []

    for idx in range(num_images):
        img_info = dataset.get_img_info(idx)
        img_id = img_info.get("image_id", img_info.get("id", idx))
        file_name = img_info.get("file_name", f"{img_id}.jpg")
        img_path = os.path.join(dataset.img_dir, file_name)

        pred_dict, fwd_ms, e2e_ms = adapter.predict_image(img_path, top_k_rels=top_k_rels)
        pred_dict["image_id"] = img_id
        pred_dict["dataset_idx"] = idx

        predictions.append(pred_dict)
        fwd_times.append(fwd_ms)
        e2e_times.append(e2e_ms)

        if (idx + 1) % 25 == 0 or (idx + 1) == num_images:
            avg_fwd = np.mean(fwd_times)
            avg_e2e = np.mean(e2e_times)
            peak_ram = get_peak_memory_mb()
            logger.info(f"Processed {idx + 1}/{num_images} images | Fwd: {avg_fwd:.1f} ms | E2E: {avg_e2e:.1f} ms | RAM: {peak_ram:.1f} MB")

    mean_fwd = float(np.mean(fwd_times))
    mean_e2e = float(np.mean(e2e_times))
    fps = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0
    peak_mem = get_peak_memory_mb()

    profile = {
        "mean_forward_latency_ms": mean_fwd,
        "mean_e2e_latency_ms": mean_e2e,
        "fps": fps,
        "peak_rss_mb": peak_mem,
    }

    if cache_path:
        torch.save({"predictions": predictions, "profile": profile}, cache_path)
        logger.info(f"Saved FCSGG predictions and profile to {cache_path}")

    return predictions, profile

def evaluate_predictions_on_view(
    predictions: List[Dict[str, Any]],
    dataset: RelationDataset,
    source_dataset_name: str,
    view_name: str,
    projector: OntologyProjector,
    k_list: List[int] = [20, 50, 100]
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth for a specified ontology view.
    """
    num_images = len(predictions)
    target_ind_to_predicates = dataset.ind_to_predicates
    num_predicates = len(target_ind_to_predicates)

    # Determine native VG150 predicate list
    with open(os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")) as f:
        vg_raw = json.load(f)
    source_ind_to_predicates = ["__background__"] + [r["name"] for r in vg_raw["rel_categories"]]

    eligible_image_indices = []
    empty_image_indices = []

    per_img_recalls_eligible = {k: [] for k in k_list}
    per_img_recalls_all = {k: [] for k in k_list}
    class_recall_collector = {k: [[] for _ in range(num_predicates)] for k in k_list}
    class_gt_counts = [0] * num_predicates
    total_gt_relations = 0

    for idx in range(num_images):
        pred_dict = predictions[idx]
        gt_dict = dataset.get_groundtruth(idx, evaluation=True)

        gt_boxes = gt_dict["boxes"].numpy()
        gt_classes = gt_dict["labels"].numpy()
        gt_rels = gt_dict["relation_tuple"].numpy()
        num_gt = len(gt_rels)
        total_gt_relations += num_gt

        if num_gt == 0:
            empty_image_indices.append(idx)
            for k in k_list:
                per_img_recalls_all[k].append(0.0)
            continue
        else:
            eligible_image_indices.append(idx)

        for r in gt_rels:
            p_cls = int(r[2])
            if 0 < p_cls < num_predicates:
                class_gt_counts[p_cls] += 1

        pred_boxes = pred_dict["boxes"].numpy()
        pred_classes = pred_dict["pred_labels"].numpy()
        obj_scores = pred_dict["pred_scores"].numpy()
        raw_pair_idxs = pred_dict["rel_pair_idxs"].numpy()
        raw_rel_scores = pred_dict["pred_rel_scores"]

        if len(raw_pair_idxs) == 0 or len(pred_boxes) == 0:
            for k in k_list:
                per_img_recalls_eligible[k].append(0.0)
                per_img_recalls_all[k].append(0.0)
            continue

        # Project relation scores if target view is not native
        if view_name != "native":
            proj_rel_scores = projector.project_prediction(
                pred_rel_scores=raw_rel_scores,
                source_dataset=source_dataset_name.upper(),
                view_name=view_name,
                source_ind_to_predicates=source_ind_to_predicates,
                target_ind_to_predicates=target_ind_to_predicates
            )
        else:
            proj_rel_scores = raw_rel_scores

        rel_scores_np = proj_rel_scores.numpy()

        # Sort predictions by top predicate confidence score
        pred_predicate_scores = rel_scores_np[:, 1:].max(axis=1) if rel_scores_np.shape[1] > 1 else np.zeros(len(rel_scores_np))
        pred_predicate_labels = 1 + rel_scores_np[:, 1:].argmax(axis=1) if rel_scores_np.shape[1] > 1 else np.ones(len(rel_scores_np), dtype=int)

        # Sort descending
        sort_order = np.argsort(-pred_predicate_scores)
        sorted_pairs = raw_pair_idxs[sort_order]
        sorted_labels = pred_predicate_labels[sort_order]
        sorted_scores = pred_predicate_scores[sort_order]

        # Prepare triplets
        pred_rels = np.column_stack((sorted_pairs, sorted_labels))
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        pred_triplets, pred_triplet_boxes, _ = _triplet(pred_rels, pred_classes, pred_boxes, sorted_scores, obj_scores)

        # Match predicted triplets with GT triplets (IoU >= 0.5 for sub & obj)
        pred_to_gt = _compute_pred_matches(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, iou_thres=0.5)

        # Compute recall at each K
        for k in k_list:
            match = reduce(np.union1d, pred_to_gt[:k]) if len(pred_to_gt) > 0 else []
            hit_count = len(match)
            img_recall = float(hit_count) / float(num_gt)
            per_img_recalls_eligible[k].append(img_recall)
            per_img_recalls_all[k].append(img_recall)

            # Class-level mean recall tracking
            recall_hit = [0] * num_predicates
            recall_cnt = [0] * num_predicates
            for r_idx in range(num_gt):
                p_cls = int(gt_rels[r_idx, 2])
                if p_cls < num_predicates:
                    recall_cnt[p_cls] += 1
            for m_idx in match:
                gt_idx = int(m_idx)
                p_cls = int(gt_rels[gt_idx, 2])
                if p_cls < num_predicates:
                    recall_hit[p_cls] += 1
            for p_cls in range(1, num_predicates):
                if recall_cnt[p_cls] > 0:
                    class_recall_collector[k][p_cls].append(float(recall_hit[p_cls]) / float(recall_cnt[p_cls]))

    # Aggregate metrics
    results = {
        "view_name": view_name,
        "total_images": num_images,
        "eligible_images": len(eligible_image_indices),
        "empty_images": len(empty_image_indices),
        "total_gt_relations": total_gt_relations,
        "recalls_eligible": {},
        "recalls_all_200": {},
        "mean_recalls": {},
        "per_predicate_recall_100": {},
        "predicate_gt_counts": {},
    }

    for k in k_list:
        el_list = per_img_recalls_eligible[k]
        results["recalls_eligible"][f"R@{k}"] = float(np.mean(el_list)) if len(el_list) > 0 else 0.0

        all_list = per_img_recalls_all[k]
        results["recalls_all_200"][f"R@{k}_all200"] = float(np.mean(all_list)) if len(all_list) > 0 else 0.0

        # Mean Recall across valid predicate classes with GT > 0
        mr_per_class = []
        for p_cls in range(1, num_predicates):
            c_hits = class_recall_collector[k][p_cls]
            if len(c_hits) > 0:
                mr_per_class.append(float(np.mean(c_hits)))
        results["mean_recalls"][f"mR@{k}"] = float(np.mean(mr_per_class)) if len(mr_per_class) > 0 else 0.0

    # Per-predicate recall at K=100
    for p_cls in range(1, num_predicates):
        p_name = target_ind_to_predicates[p_cls]
        c_hits = class_recall_collector[100][p_cls]
        results["per_predicate_recall_100"][p_name] = float(np.mean(c_hits)) if len(c_hits) > 0 else 0.0
        results["predicate_gt_counts"][p_name] = class_gt_counts[p_cls]

    return results

def evaluate_detection_mAP(predictions: List[Dict[str, Any]], dataset: RelationDataset) -> float:
    """
    Computes standard object detection mAP@50 using COCOeval.
    """
    images = []
    anns = []
    ann_id = 0

    for idx, pred in enumerate(predictions):
        img_info = dataset.get_img_info(idx)
        images.append({
            "id": idx,
            "width": int(pred["orig_size"][1]),
            "height": int(pred["orig_size"][0]),
        })

        gt_dict = dataset.get_groundtruth(idx, evaluation=True)
        gt_boxes = gt_dict["boxes"].numpy()
        gt_labels = gt_dict["labels"].numpy()

        for box, lbl in zip(gt_boxes, gt_labels):
            w = max(0.0, box[2] - box[0])
            h = max(0.0, box[3] - box[1])
            anns.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(lbl),
                "bbox": [float(box[0]), float(box[1]), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i, "name": n} for i, n in enumerate(dataset.ind_to_classes) if n != "__background__"]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": anns, "categories": categories}
    coco_gt.createIndex()

    dt_anns = []
    for idx, pred in enumerate(predictions):
        boxes = pred["boxes"].numpy()
        scores = pred["pred_scores"].numpy()
        labels = pred["pred_labels"].numpy()
        for b, s, l in zip(boxes, scores, labels):
            w = max(0.0, b[2] - b[0])
            h = max(0.0, b[3] - b[1])
            dt_anns.append({
                "image_id": idx,
                "category_id": int(l),
                "bbox": [float(b[0]), float(b[1]), float(w), float(h)],
                "score": float(s),
            })

    if not dt_anns:
        return 0.0

    coco_dt = coco_gt.loadRes(dt_anns)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(range(len(predictions)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[1])  # mAP@50

def run_fcsgg_benchmark(smoke_test: bool = False):
    """Executes FCSGG benchmark suite."""
    logger.info("=" * 80)
    logger.info(f"STARTING FCSGG BENCHMARK SUITE | Mode: {'SMOKE TEST (3 IMAGES)' if smoke_test else 'FULL EVALUATION (200 IMAGES)'}")
    logger.info("=" * 80)

    max_imgs = 3 if smoke_test else 200
    ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints/FCSGG/HRNetW32-1S.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"FCSGG checkpoint not found: {ckpt_path}")

    adapter = FCSGGAdapter(checkpoint_path=ckpt_path)
    projector = OntologyProjector(os.path.join(WORKSPACE_ROOT, "evaluation_sets/evaluation_ontology.json"))

    img_dir = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/VG150")
    views = [
        ("strict", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_strict.coco.json")),
        ("extended", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_extended.coco.json")),
        ("tri_dataset", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_tri_dataset.coco.json")),
        ("native", os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")),
    ]

    native_dataset = RelationDataset(
        annotation_file=views[-1][1],
        img_dir=img_dir,
        filter_empty_rels=False,
        filter_duplicate_rels=False,
        filter_non_overlap=False
    )

    raw_cache = os.path.join(CACHE_DIR, f"raw_preds_VG150_fcsgg_{'smoke' if smoke_test else 'full'}.pt")
    preds, profile = run_fcsgg_inference(
        adapter=adapter,
        dataset=native_dataset,
        max_images=max_imgs,
        top_k_rels=100,
        cache_path=raw_cache
    )

    logger.info("Evaluating Object Detection mAP@50...")
    det_map50 = evaluate_detection_mAP(preds, native_dataset)
    logger.info(f"Object Detection mAP@50 (CenterNet / HRNetW32): {det_map50:.4f}")

    benchmark_records = []
    for view_name, ann_path in views:
        logger.info(f"Evaluating View={view_name} ({ann_path})...")
        view_dataset = RelationDataset(
            annotation_file=ann_path,
            img_dir=img_dir,
            filter_empty_rels=False,
            filter_duplicate_rels=False,
            filter_non_overlap=False
        )

        res = evaluate_predictions_on_view(
            predictions=preds,
            dataset=view_dataset,
            source_dataset_name="VG150",
            view_name=view_name,
            projector=projector,
            k_list=[20, 50, 100]
        )

        record = {
            "model": "FCSGG",
            "checkpoint": ckpt_path,
            "source_dataset": "VG150",
            "eval_dataset": "VG150",
            "ontology_view": view_name,
            "task": "SGDet",
            "status": "COMPLETED",
            "expected_images": max_imgs,
            "processed_images": len(preds),
            "eligible_images": res["eligible_images"],
            "empty_images": res["empty_images"],
            "total_gt_relations": res["total_gt_relations"],
            "detection_mAP50": det_map50,
            "R@20": res["recalls_eligible"]["R@20"],
            "R@50": res["recalls_eligible"]["R@50"],
            "R@100": res["recalls_eligible"]["R@100"],
            "R@20_all200": res["recalls_all_200"]["R@20_all200"],
            "R@50_all200": res["recalls_all_200"]["R@50_all200"],
            "R@100_all200": res["recalls_all_200"]["R@100_all200"],
            "mR@20": res["mean_recalls"]["mR@20"],
            "mR@50": res["mean_recalls"]["mR@50"],
            "mR@100": res["mean_recalls"]["mR@100"],
            "mean_forward_latency_ms": profile["mean_forward_latency_ms"],
            "mean_e2e_latency_ms": profile["mean_e2e_latency_ms"],
            "fps": profile["fps"],
            "peak_rss_mb": profile["peak_rss_mb"],
            "per_predicate_recall_100": res["per_predicate_recall_100"],
            "predicate_gt_counts": res["predicate_gt_counts"],
        }
        benchmark_records.append(record)

        logger.info(
            f"[FCSGG | {view_name.upper()}] R@20: {record['R@20']:.4f} | "
            f"R@50: {record['R@50']:.4f} | R@100: {record['R@100']:.4f} | "
            f"mR@50: {record['mR@50']:.4f} | mR@100: {record['mR@100']:.4f} | "
            f"All-200 R@50: {record['R@50_all200']:.4f} | "
            f"Eligible: {res['eligible_images']}/{max_imgs}"
        )

    # Save outputs
    metrics_json = os.path.join(LOG_DIR, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
    logger.info(f"Saved metrics to {metrics_json}")

    metrics_csv = os.path.join(LOG_DIR, "metrics.csv")
    csv_fields = [
        "model", "checkpoint", "source_dataset", "eval_dataset", "ontology_view", "task",
        "status", "expected_images", "processed_images", "eligible_images", "empty_images",
        "total_gt_relations", "detection_mAP50", "R@20", "R@50", "R@100", "R@20_all200",
        "R@50_all200", "R@100_all200", "mR@20", "mR@50", "mR@100",
        "mean_forward_latency_ms", "mean_e2e_latency_ms", "fps", "peak_rss_mb"
    ]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in benchmark_records:
            row = {k: r[k] for k in csv_fields}
            writer.writerow(row)
    logger.info(f"Saved metrics CSV to {metrics_csv}")

    run_config = {
        "model": "FCSGG",
        "checkpoint": ckpt_path,
        "config": "model_repos/fcsgg/configs/FCSGG_HRNet_W32_2xDownRAF_512x512_MS.yaml",
        "device": "cpu",
        "hardware": "Intel(R) Core(TM) Ultra 5 125H (18 threads)",
        "os": "Windows 11",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "SMOKE_TEST" if smoke_test else "FULL_EVALUATION",
        "max_images": max_imgs,
        "views": [v[0] for v in views],
        "profile": profile,
    }
    config_json = os.path.join(LOG_DIR, "run_config.json")
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    logger.info(f"Saved run config to {config_json}")

    logger.info("Evaluation suite finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FCSGG benchmark suite")
    parser.add_argument("--smoke-test", action="store_true", help="Run 3-image smoke test")
    args = parser.parse_args()

    run_fcsgg_benchmark(smoke_test=args.smoke_test)
