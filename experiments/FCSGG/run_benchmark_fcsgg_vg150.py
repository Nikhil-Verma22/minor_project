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
import cv2
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
from experiments.FCSGG.fcsgg_adapter import FCSGGAdapter

LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/FCSGG")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/FCSGG")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("FCSGGVG150Benchmark")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(os.path.join(LOG_DIR, "vg150_run.log"), mode="a", encoding="utf-8")
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def get_peak_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_fcsgg_vg150_inference(
    adapter: Optional[FCSGGAdapter],
    dataset: RelationDataset,
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Executes FCSGG inference on the VG150 evaluation dataset or loads cached predictions.
    Since FCSGG is natively trained on VG150, predictions map 1:1 with native VG150 ontology.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached FCSGG VG150 predictions from {cache_path}...")
        cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached_data["predictions"][:max_images], cached_data["profile"]

    # Fallback to archive cache if present
    archive_cache = os.path.join(WORKSPACE_ROOT, "archive/batch_testing/cache/raw_preds_VG150_fcsgg_full.pt")
    if os.path.exists(archive_cache):
        logger.info(f"Loading cached FCSGG VG150 predictions from archive: {archive_cache}...")
        cached_data = torch.load(archive_cache, map_location="cpu", weights_only=False)
        return cached_data["predictions"][:max_images], cached_data["profile"]

    if adapter is None:
        raise ValueError("Adapter must be provided when running fresh inference.")

    num_images = min(max_images, len(dataset))
    logger.info(f"Running FCSGG inference across {num_images} VG150 images...")

    predictions = []
    fwd_times = []
    e2e_times = []

    for idx in range(num_images):
        img_info = dataset.get_img_info(idx)
        img_id = img_info.get("image_id", img_info.get("id", idx))
        file_name = img_info.get("file_name", f"{img_id}.jpg")
        img_path = os.path.join(dataset.img_dir, file_name)

        if not os.path.exists(img_path):
            img_path = dataset.filenames[idx]

        pred_raw, fwd_ms, e2e_ms = adapter.predict_image(img_path, top_k_rels=top_k_rels)

        pred_dict = {
            "image_id": img_id,
            "dataset_idx": idx,
            "orig_size": pred_raw["orig_size"],
            "boxes": pred_raw["boxes"],
            "pred_scores": pred_raw["pred_scores"],
            "pred_labels": pred_raw["pred_labels"],
            "rel_pair_idxs": pred_raw["rel_pair_idxs"],
            "pred_rel_scores": pred_raw["pred_rel_scores"],
            "pred_rel_labels": pred_raw["pred_rel_labels"],
        }

        predictions.append(pred_dict)
        fwd_times.append(fwd_ms)
        e2e_times.append(e2e_ms)

        if (idx + 1) % 25 == 0 or (idx + 1) == num_images:
            avg_fwd = np.mean(fwd_times)
            avg_e2e = np.mean(e2e_times)
            peak_ram = get_peak_memory_mb()
            logger.info(
                f"Processed {idx + 1:3d}/{num_images:3d} images | "
                f"Fwd: {avg_fwd:.1f} ms | E2E: {avg_e2e:.1f} ms | RAM: {peak_ram:.1f} MB"
            )

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
        logger.info(f"Saved FCSGG VG150 predictions cache to {cache_path}")

    return predictions, profile


def evaluate_detection_mAP(predictions: List[Dict[str, Any]], dataset: RelationDataset) -> Tuple[float, float]:
    """
    Computes standard object detection mAP@50 and F1@50 on VG150 using COCOeval.
    """
    logger.info("Computing detection mAP@50 and F1@50 for FCSGG on VG150...")
    images = []
    anns = []
    ann_id = 0

    for idx in range(len(predictions)):
        img_info = dataset.get_img_info(idx)
        images.append({
            "id": idx,
            "width": int(img_info.get("width", 500)),
            "height": int(img_info.get("height", 500)),
        })
        gt = dataset.get_groundtruth(idx, evaluation=True)
        boxes = gt["boxes"].numpy()
        labels = gt["labels"].numpy()
        for box, label in zip(boxes, labels):
            w = max(0.0, float(box[2] - box[0]))
            h = max(0.0, float(box[3] - box[1]))
            anns.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(label),
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
            if int(l) > 0:
                w = max(0.0, float(b[2] - b[0]))
                h = max(0.0, float(b[3] - b[1]))
                dt_anns.append({
                    "image_id": idx,
                    "category_id": int(l),
                    "bbox": [float(b[0]), float(b[1]), float(w), float(h)],
                    "score": float(s),
                })

    if not dt_anns:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(dt_anns)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(range(len(predictions)))
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50 = float(coco_eval.stats[0])
    precision_arr = coco_eval.eval["precision"][0, :, :, 0, 2]
    valid_prec = precision_arr[precision_arr > -1]
    mean_p = float(np.mean(valid_prec)) if len(valid_prec) > 0 else 0.0

    recall_arr = coco_eval.eval["recall"][0, :, 0, 2]
    valid_rec = recall_arr[recall_arr > -1]
    mean_r = float(np.mean(valid_rec)) if len(valid_rec) > 0 else 0.0

    f1_50 = 2 * (mean_p * mean_r) / (mean_p + mean_r + 1e-8) if (mean_p + mean_r) > 0 else 0.0
    return map_50, f1_50


def evaluate_sgg_recalls(
    predictions: List[Dict[str, Any]],
    dataset: RelationDataset,
    k_list: List[int] = [50, 100]
) -> Dict[str, Any]:
    """
    Computes SGG Recall@K and Mean Recall@K for FCSGG on VG150.
    """
    num_images = len(predictions)
    target_ind_to_predicates = dataset.ind_to_predicates
    num_predicates = len(target_ind_to_predicates)

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
        rel_scores_np = pred_dict["pred_rel_scores"].numpy()

        if len(raw_pair_idxs) == 0 or len(pred_boxes) == 0:
            for k in k_list:
                per_img_recalls_eligible[k].append(0.0)
                per_img_recalls_all[k].append(0.0)
            continue

        # Extract top predicate per pair and rank by triplet score
        pred_predicate_scores = rel_scores_np[:, 1:].max(axis=1) if rel_scores_np.shape[1] > 1 else np.zeros(len(rel_scores_np))
        pred_predicate_labels = 1 + rel_scores_np[:, 1:].argmax(axis=1) if rel_scores_np.shape[1] > 1 else np.ones(len(rel_scores_np), dtype=int)

        # Pair confidence = obj_scores[subj] * obj_scores[obj] * pred_predicate_scores
        s_idxs = raw_pair_idxs[:, 0]
        o_idxs = raw_pair_idxs[:, 1]
        pair_conf = obj_scores[s_idxs] * obj_scores[o_idxs] * pred_predicate_scores

        sort_order = np.argsort(-pair_conf)
        sorted_pairs = raw_pair_idxs[sort_order]
        sorted_labels = pred_predicate_labels[sort_order]
        sorted_scores = pair_conf[sort_order]

        pred_rels = np.column_stack((sorted_pairs, sorted_labels))
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        pred_triplets, pred_triplet_boxes, _ = _triplet(pred_rels, pred_classes, pred_boxes, sorted_scores, obj_scores)

        pred_to_gt = _compute_pred_matches(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, iou_thres=0.5)

        for k in k_list:
            match = reduce(np.union1d, pred_to_gt[:k]) if len(pred_to_gt) > 0 else []
            hit_count = len(match)
            img_recall = float(hit_count) / float(num_gt)
            per_img_recalls_eligible[k].append(img_recall)
            per_img_recalls_all[k].append(img_recall)

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

    results = {
        "total_images": num_images,
        "eligible_images": len(eligible_image_indices),
        "empty_images": len(empty_image_indices),
        "total_gt_relations": total_gt_relations,
        "recalls_eligible": {},
        "recalls_all_200": {},
        "mean_recalls": {},
        "per_predicate_recall_100": {},
    }

    for k in k_list:
        results["recalls_eligible"][f"R@{k}"] = float(np.mean(per_img_recalls_eligible[k])) if per_img_recalls_eligible[k] else 0.0
        results["recalls_all_200"][f"R@{k}_all200"] = float(np.mean(per_img_recalls_all[k])) if per_img_recalls_all[k] else 0.0

        valid_m_recalls = []
        for p_cls in range(1, num_predicates):
            if class_recall_collector[k][p_cls]:
                valid_m_recalls.append(np.mean(class_recall_collector[k][p_cls]))
        results["mean_recalls"][f"mR@{k}"] = float(np.mean(valid_m_recalls)) if valid_m_recalls else 0.0

    for p_cls in range(1, num_predicates):
        p_name = target_ind_to_predicates[p_cls]
        gt_cnt = class_gt_counts[p_cls]
        if class_recall_collector[100][p_cls]:
            rec_val = float(np.mean(class_recall_collector[100][p_cls]))
        else:
            rec_val = 0.0 if gt_cnt > 0 else None
        results["per_predicate_recall_100"][p_name] = {
            "recall": rec_val,
            "gt_count": gt_cnt
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Fully Convolutional SGG (FCSGG) Benchmark on Visual Genome (VG150)")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Starting Fully Convolutional SGG (FCSGG) Benchmark Pipeline on VG150 Dataset")
    logger.info(f"Max Images: {args.max_images} | Device: {args.device}")
    logger.info("=" * 80)

    annotation_file = os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")
    img_dir = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/VG150")

    vg_dataset = RelationDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        filter_empty_rels=False,
        filter_duplicate_rels=False,
        filter_non_overlap=False
    )
    logger.info(f"Loaded VG150 evaluation dataset: {len(vg_dataset)} images")

    cache_path = os.path.join(CACHE_DIR, "raw_preds_VG150_full.pt")
    adapter = None
    if not os.path.exists(cache_path):
        adapter = FCSGGAdapter(device=args.device)

    predictions, profile = run_fcsgg_vg150_inference(
        adapter=adapter,
        dataset=vg_dataset,
        max_images=args.max_images,
        top_k_rels=100,
        cache_path=cache_path
    )

    # Compute object detection metrics
    map_50, f1_50 = evaluate_detection_mAP(predictions, vg_dataset)
    logger.info(f"FCSGG Object Detection on VG150: mAP@50 = {map_50:.4f} | F1@50 = {f1_50:.4f}")

    # Compute SGG recall metrics
    sgg_metrics = evaluate_sgg_recalls(predictions, vg_dataset, k_list=[50, 100])

    r50 = sgg_metrics["recalls_eligible"]["R@50"]
    r100 = sgg_metrics["recalls_eligible"]["R@100"]
    mr50 = sgg_metrics["mean_recalls"]["mR@50"]
    mr100 = sgg_metrics["mean_recalls"]["mR@100"]
    latency_ms = profile["mean_e2e_latency_ms"]
    fps = profile["fps"]
    speed_str = f"{latency_ms:.1f} ms ({fps:.2f} FPS)"

    logger.info("=" * 80)
    logger.info("FINAL BENCHMARK SCORES FOR FCSGG ON VG150:")
    logger.info(f"R@50:            {r50*100:.2f}%")
    logger.info(f"R@100:           {r100*100:.2f}%")
    logger.info(f"mR@50:           {mr50*100:.2f}%")
    logger.info(f"mR@100:          {mr100*100:.2f}%")
    logger.info(f"F1@50:           {f1_50*100:.2f}% (or {f1_50:.4f})")
    logger.info(f"Detection mAP@50: {map_50*100:.2f}% (or {map_50:.4f})")
    logger.info(f"Speed:           {speed_str}")
    logger.info("=" * 80)

    # Save metrics JSON
    out_dict = {
        "model": "Fully Convolutional SGG (FCSGG)",
        "dataset": "VG150 (Visual Genome)",
        "num_images": len(predictions),
        "scores": {
            "R@50": r50,
            "R@100": r100,
            "mR@50": mr50,
            "mR@100": mr100,
            "F1@50": f1_50,
            "Detection_mAP@50": map_50,
            "Speed_latency_ms": latency_ms,
            "Speed_fps": fps,
            "Speed": speed_str,
        },
        "profile": profile,
        "sgg_metrics": sgg_metrics,
    }

    metrics_json_path = os.path.join(LOG_DIR, "vg150_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    logger.info(f"Saved FCSGG VG150 metrics to {metrics_json_path}")

    # Save metrics CSV
    metrics_csv_path = os.path.join(LOG_DIR, "vg150_metrics.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "R@50", "R@100", "mR@50", "mR@100", "F1@50", "Detection mAP@50", "Speed"])
        writer.writerow([
            "Fully Convolutional SGG (FCSGG)",
            "VG150",
            f"{r50*100:.2f}%",
            f"{r100*100:.2f}%",
            f"{mr50*100:.2f}%",
            f"{mr100*100:.2f}%",
            f"{f1_50*100:.2f}%",
            f"{map_50*100:.2f}%",
            speed_str
        ])
    logger.info(f"Saved FCSGG VG150 metrics CSV to {metrics_csv_path}")

    # Print requested table format
    print("\n" + "=" * 80)
    print("REQUESTED BENCHMARK OUTPUT TABLE:")
    print("R@50\tR@100\tmR@50\tmR@100\tF1@50\tDetection mAP@50\tSpeed")
    print(f"{r50*100:.2f}%\t{r100*100:.2f}%\t{mr50*100:.2f}%\t{mr100*100:.2f}%\t{f1_50*100:.2f}%\t{map_50*100:.2f}%\t{speed_str}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
