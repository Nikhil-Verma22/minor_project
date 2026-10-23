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
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

WORKSPACE_ROOT = "d:/Minor"
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/SGG-Benchmark"))

from sgg_benchmark.data.datasets.data import RelationDataset
from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
)
from experiments.REACT.test_single_image_inference import letterbox
from experiments.FastContextualSGG.run_benchmark_fast_contextual_openimages import build_ontology_mappings

LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/REACTPP")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/REACTPP")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("REACTPPOpenImagesBenchmark")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(os.path.join(LOG_DIR, "openimages_run.log"), mode="a", encoding="utf-8")
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def get_peak_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def undo_letterbox(boxes: np.ndarray, gain: float, pad_w: float, pad_h: float, orig_w: int, orig_h: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_w) / gain
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_h) / gain
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, orig_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, orig_h)
    return b


def run_reactpp_openimages_inference(
    onnx_path: str,
    dataset: RelationDataset,
    vg_to_oid_obj: Dict[int, int],
    vg_to_oid_pred: Dict[int, int],
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached REACT++ OpenImages predictions from {cache_path}...")
        cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached_data["predictions"][:max_images], cached_data["profile"]

    logger.info(f"Loading ONNX session from {onnx_path}...")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logger.info(f"ONNX session initialized successfully. Input: {input_name}")

    # Warmup
    logger.info("Warming up ONNX model (3 iterations)...")
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    for _ in range(3):
        session.run(None, {input_name: dummy})

    num_images = min(max_images, len(dataset))
    logger.info(f"Running REACT++ (YOLO12m ONNX) inference across {num_images} OpenImages images...")

    predictions = []
    fwd_times = []
    e2e_times = []
    cand_pair_counts = []

    for idx in range(num_images):
        t0_e2e = time.perf_counter()
        img_info = dataset.get_img_info(idx)
        img_id = img_info.get("image_id", img_info.get("id", idx))
        file_name = img_info.get("file_name", f"{img_id}.jpg")
        img_path = os.path.join(dataset.img_dir, file_name)

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image from: {img_path}")
        orig_h, orig_w = bgr.shape[:2]

        padded, gain, pad_w, pad_h = letterbox(bgr, 640)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

        t0_fwd = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        t1_fwd = time.perf_counter()

        boxes_raw = outputs[0].copy() if len(outputs) > 0 else np.zeros((0, 6), dtype=np.float32)
        rels_raw = outputs[1].copy() if len(outputs) > 1 else np.zeros((0, 5), dtype=np.float32)

        if len(boxes_raw) > 0:
            boxes_raw = undo_letterbox(boxes_raw, gain, pad_w, pad_h, orig_w, orig_h)

        if len(rels_raw) > top_k_rels:
            order = np.argsort(rels_raw[:, 3])[::-1][:top_k_rels]
            rels_raw = rels_raw[order]

        boxes = boxes_raw[:, :4] if len(boxes_raw) > 0 else np.zeros((0, 4), dtype=np.float32)
        raw_labels = boxes_raw[:, 4].astype(np.int64) if len(boxes_raw) > 0 else np.zeros(0, dtype=np.int64)
        scores = boxes_raw[:, 5].astype(np.float32) if len(boxes_raw) > 0 else np.zeros(0, dtype=np.float32)

        # Map object labels: VG150 ID -> OpenImages ID (0 if unmapped)
        mapped_labels = np.array([vg_to_oid_obj.get(int(l), 0) for l in raw_labels], dtype=np.int64)

        # Build 32-column relation score matrix (col 0: background, cols 1..31: OpenImages predicates)
        num_pairs = len(rels_raw)
        mapped_rel_scores = np.zeros((num_pairs, 32), dtype=np.float32)
        pair_idxs = np.zeros((num_pairs, 2), dtype=np.int64)

        if num_pairs > 0:
            pair_idxs[:, 0] = rels_raw[:, 0].astype(np.int64)
            pair_idxs[:, 1] = rels_raw[:, 1].astype(np.int64)
            v_preds = rels_raw[:, 2].astype(np.int64)
            tri_scores = rels_raw[:, 3].astype(np.float32)

            for i, (vp, ts) in enumerate(zip(v_preds, tri_scores)):
                oid_p = vg_to_oid_pred.get(int(vp), 0)
                if 1 <= oid_p < 32:
                    mapped_rel_scores[i, oid_p] = float(ts)

        t1_e2e = time.perf_counter()
        fwd_times.append((t1_fwd - t0_fwd) * 1000.0)
        e2e_times.append((t1_e2e - t0_e2e) * 1000.0)
        cand_pair_counts.append(num_pairs)

        pred_dict = {
            "image_id": img_id,
            "dataset_idx": idx,
            "orig_size": (orig_h, orig_w),
            "boxes": torch.from_numpy(boxes),
            "pred_scores": torch.from_numpy(scores),
            "pred_labels": torch.from_numpy(mapped_labels),
            "raw_vg_labels": torch.from_numpy(raw_labels),
            "rel_pair_idxs": torch.from_numpy(pair_idxs),
            "pred_rel_scores": torch.from_numpy(mapped_rel_scores),
        }
        predictions.append(pred_dict)

        if (idx + 1) % 25 == 0 or (idx + 1) == num_images:
            logger.info(
                f"[REACT++] Processed {idx + 1:3d}/{num_images:3d} images | "
                f"E2E: {np.mean(e2e_times[-25:]):.1f} ms (Fwd: {np.mean(fwd_times[-25:]):.1f} ms) | "
                f"RAM: {get_peak_memory_mb():.1f} MB"
            )

    mean_fwd = float(np.mean(fwd_times))
    mean_e2e = float(np.mean(e2e_times))
    fps = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0
    peak_mem = get_peak_memory_mb()

    profile = {
        "model": "REACT++ (YOLO12m ONNX)",
        "mean_fwd_latency_ms": mean_fwd,
        "mean_e2e_latency_ms": mean_e2e,
        "fps": fps,
        "peak_rss_mb": peak_mem,
        "mean_candidate_pairs": float(np.mean(cand_pair_counts)),
    }

    if cache_path:
        torch.save({"predictions": predictions, "profile": profile}, cache_path)
        logger.info(f"Saved REACT++ OpenImages predictions and profile to {cache_path}")

    return predictions, profile


def evaluate_reactpp_openimages_recalls(
    predictions: List[Dict[str, Any]],
    dataset: RelationDataset,
    k_list: List[int] = [50, 100],
    exclude_predicate: Optional[int] = None
) -> Dict[str, Any]:
    num_images = len(predictions)
    target_ind_to_predicates = dataset.ind_to_predicates
    num_predicates = len(target_ind_to_predicates)

    eligible_image_indices = []
    empty_image_indices = []

    per_img_recalls_eligible = {k: [] for k in k_list}
    per_img_recalls_all = {k: [] for k in k_list}
    class_recall_collector = {k: [[] for _ in range(num_predicates)] for k in k_list}
    class_gt_counts = [0] * num_predicates
    class_hit_counts = {k: [0] * num_predicates for k in k_list}
    total_gt_relations = 0
    total_hits = {k: 0 for k in k_list}

    for idx in range(num_images):
        pred_dict = predictions[idx]
        gt_dict = dataset.get_groundtruth(idx, evaluation=True)

        gt_boxes = gt_dict["boxes"].numpy()
        gt_classes = gt_dict["labels"].numpy()
        gt_rels = gt_dict["relation_tuple"].numpy()

        if exclude_predicate is not None and len(gt_rels) > 0:
            mask = gt_rels[:, 2] != exclude_predicate
            gt_rels = gt_rels[mask]

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

        # Extract top predicate per pair
        pred_predicate_scores = rel_scores_np[:, 1:].max(axis=1) if rel_scores_np.shape[1] > 1 else np.zeros(len(rel_scores_np))
        pred_predicate_labels = 1 + rel_scores_np[:, 1:].argmax(axis=1) if rel_scores_np.shape[1] > 1 else np.ones(len(rel_scores_np), dtype=int)

        # NOTE ON SCORING (TASK 1 CORRECTION):
        # In REACT++ ONNX output, rels_raw[:, 3] already contains the model's precomputed triplet confidence:
        #   triplet_score = (obj_scores[subj] * obj_scores[obj] * rel_score) ** (1/3).
        # mapped_rel_scores was populated directly from rels_raw[:, 3].
        # Downstream ranking must rank directly by pred_predicate_scores.
        # Multiplying by obj_scores[subj] * obj_scores[obj] again double-counts detection confidences
        # and distorts candidate ranking. Direct ranking matches native REACT++ (PSG and VG150).
        sort_order = np.argsort(-pred_predicate_scores)
        sorted_pairs = raw_pair_idxs[sort_order]
        sorted_labels = pred_predicate_labels[sort_order]
        sorted_scores = pred_predicate_scores[sort_order]

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
                valid_m_recalls.append(float(np.mean(class_recall_collector[k][p_cls])))
        results["mean_recalls"][f"mR@{k}"] = float(np.mean(valid_m_recalls)) if valid_m_recalls else 0.0

    for p_cls in range(1, num_predicates):
        p_name = target_ind_to_predicates[p_cls]
        vals = class_recall_collector[100][p_cls]
        results["per_predicate_recall_100"][p_name] = {
            "recall": float(np.mean(vals)) if vals else None,
            "gt_count": class_gt_counts[p_cls]
        }

    return results


def evaluate_reactpp_openimages_detection(
    predictions: List[Dict[str, Any]],
    dataset: RelationDataset
) -> Tuple[float, float]:
    """Computes COCO detection mAP@50 and F1@50."""
    logger.info("Computing detection mAP@50 and F1@50 for REACT++ on OpenImages V6...")
    images = []
    anns = []
    ann_id = 1

    for idx in range(len(predictions)):
        img_info = dataset.get_img_info(idx)
        images.append({
            "id": idx,
            "width": img_info.get("width", 640),
            "height": img_info.get("height", 640),
            "file_name": img_info.get("file_name", f"{idx}.jpg")
        })
        gt_dict = dataset.get_groundtruth(idx, evaluation=True)
        boxes = gt_dict["boxes"].numpy()
        labels = gt_dict["labels"].numpy()
        for b, l in zip(boxes, labels):
            w = max(0.0, float(b[2] - b[0]))
            h = max(0.0, float(b[3] - b[1]))
            anns.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(l),
                "bbox": [float(b[0]), float(b[1]), float(w), float(h)],
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


def main():
    parser = argparse.ArgumentParser(description="REACT++ Benchmark on OpenImages V6 Dataset")
    parser.add_argument("--onnx_path", type=str, default="checkpoints/REACT/react_pp_vg150_yolo12m.onnx", help="Path to ONNX weights")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images to evaluate")
    args = parser.parse_args()

    onnx_path = os.path.join(WORKSPACE_ROOT, args.onnx_path) if not os.path.isabs(args.onnx_path) else args.onnx_path

    logger.info("=" * 80)
    logger.info("Starting REACT++ (YOLO12m ONNX) Benchmark Pipeline on OpenImages V6 Dataset")
    logger.info(f"Model: {onnx_path} | Max Images: {args.max_images}")
    logger.info("=" * 80)

    annotation_file = os.path.join(WORKSPACE_ROOT, "evaluation_sets/OpenImages_200_annotations.coco.json")
    img_dir = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/OpenImages")

    oid_dataset = RelationDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        filter_empty_rels=False,
        filter_duplicate_rels=False,
        filter_non_overlap=False
    )
    logger.info(f"Loaded OpenImages V6 evaluation dataset: {len(oid_dataset)} images")

    vg_to_oid_obj, vg_to_oid_pred, _ = build_ontology_mappings()
    logger.info(f"Initialized ontology mappings: {len(vg_to_oid_obj)} object classes, {len(vg_to_oid_pred)} predicates")

    mode_str = "smoke" if args.max_images <= 5 else "full"
    cache_file = os.path.join(CACHE_DIR, f"raw_preds_OpenImages_reactpp_{mode_str}.pt")

    predictions, profile = run_reactpp_openimages_inference(
        onnx_path=onnx_path,
        dataset=oid_dataset,
        vg_to_oid_obj=vg_to_oid_obj,
        vg_to_oid_pred=vg_to_oid_pred,
        max_images=args.max_images,
        top_k_rels=100,
        cache_path=cache_file
    )

    det_map50, det_f1_50 = evaluate_reactpp_openimages_detection(predictions, oid_dataset)
    logger.info(f"REACT++ Object Detection on OpenImages V6: mAP@50 = {det_map50:.4f} | F1@50 = {det_f1_50:.4f}")

    eval_results_a = evaluate_reactpp_openimages_recalls(predictions, oid_dataset, k_list=[50, 100], exclude_predicate=None)
    eval_results_b = evaluate_reactpp_openimages_recalls(predictions, oid_dataset, k_list=[50, 100], exclude_predicate=27)

    r50_a = eval_results_a["recalls_eligible"]["R@50"]
    r100_a = eval_results_a["recalls_eligible"]["R@100"]
    mr50_a = eval_results_a["mean_recalls"]["mR@50"]
    mr100_a = eval_results_a["mean_recalls"]["mR@100"]

    r50_b = eval_results_b["recalls_eligible"]["R@50"]
    r100_b = eval_results_b["recalls_eligible"]["R@100"]
    mr50_b = eval_results_b["mean_recalls"]["mR@50"]
    mr100_b = eval_results_b["mean_recalls"]["mR@100"]

    speed_ms = profile["mean_e2e_latency_ms"]
    fps = profile["fps"]
    speed_str = f"{speed_ms:.1f} ms ({fps:.2f} FPS)"

    logger.info("=" * 80)
    logger.info("FINAL BENCHMARK SCORES FOR REACT++ ON OPENIMAGES V6:")
    logger.info("--- Protocol A: Original Annotations (866 GT, 200 Images) ---")
    logger.info(f"R@50:            {r50_a * 100:.2f}%")
    logger.info(f"R@100:           {r100_a * 100:.2f}%")
    logger.info(f"mR@50:           {mr50_a * 100:.2f}%")
    logger.info(f"mR@100:          {mr100_a * 100:.2f}%")
    logger.info("--- Protocol B: Physical Relations Only (146 GT, 58 Eligible Images, Excl 'is') ---")
    logger.info(f"R@50:            {r50_b * 100:.2f}%")
    logger.info(f"R@100:           {r100_b * 100:.2f}%")
    logger.info(f"mR@50:           {mr50_b * 100:.2f}%")
    logger.info(f"mR@100:          {mr100_b * 100:.2f}%")
    logger.info("--- Detection & Speed ---")
    logger.info(f"Detection H(AP,AR)@50: {det_f1_50 * 100:.2f}% (or {det_f1_50:.4f})")
    logger.info(f"Detection mAP@50:      {det_map50 * 100:.2f}% (or {det_map50:.4f})")
    logger.info(f"Speed:                 {speed_str}")
    logger.info("=" * 80)

    # Save to JSON
    json_path = os.path.join(LOG_DIR, "openimages_metrics.json")
    final_payload = {
        "model": "REACT++",
        "variant": "YOLO12m ONNX",
        "dataset": "OpenImages V6",
        "num_images": len(predictions),
        "scores": {
            "R@50": r50_a,
            "R@100": r100_a,
            "mR@50": mr50_a,
            "mR@100": mr100_a,
            "F1@50": det_f1_50,
            "Detection_mAP@50": det_map50,
            "Speed_latency_ms": speed_ms,
            "Speed_fps": fps,
            "Speed": speed_str,
            "protocol_b": {
                "description": "Physical pairwise relations only (excluding predicate 27 'is', 146 relations across 58 eligible images)",
                "R@50": r50_b,
                "R@100": r100_b,
                "mR@50": mr50_b,
                "mR@100": mr100_b
            }
        },
        "profile": profile,
        "sgg_metrics": eval_results_a,
        "sgg_metrics_protocol_b": eval_results_b,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Saved REACT++ OpenImages metrics to {json_path}")

    # Save to CSV
    csv_path = os.path.join(LOG_DIR, "openimages_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "Protocol", "R@50", "R@100", "mR@50", "mR@100", "F1@50", "Detection mAP@50", "Speed"])
        writer.writerow([
            "REACT++ (YOLO12m ONNX)",
            "OpenImages V6",
            "Protocol A (Original 866 GT)",
            f"{r50_a * 100:.2f}%",
            f"{r100_a * 100:.2f}%",
            f"{mr50_a * 100:.2f}%",
            f"{mr100_a * 100:.2f}%",
            f"{det_f1_50 * 100:.2f}%",
            f"{det_map50 * 100:.2f}%",
            speed_str,
        ])
        writer.writerow([
            "REACT++ (YOLO12m ONNX)",
            "OpenImages V6",
            "Protocol B (Physical 146 GT)",
            f"{r50_b * 100:.2f}%",
            f"{r100_b * 100:.2f}%",
            f"{mr50_b * 100:.2f}%",
            f"{mr100_b * 100:.2f}%",
            f"{det_f1_50 * 100:.2f}%",
            f"{det_map50 * 100:.2f}%",
            speed_str,
        ])
    logger.info(f"Saved REACT++ OpenImages metrics CSV to {csv_path}")

    print("\n" + "=" * 90)
    print("REACT++ ON OPENIMAGES V6 BENCHMARK TABLE:")
    print("Protocol\tR@50\tR@100\tmR@50\tmR@100\tDet. H(AP,AR)\tDet. mAP@50\tSpeed")
    print(f"Protocol A\t{r50_a * 100:.2f}%\t{r100_a * 100:.2f}%\t{mr50_a * 100:.2f}%\t{mr100_a * 100:.2f}%\t{det_f1_50 * 100:.2f}%\t{det_map50 * 100:.2f}%\t{speed_str}")
    print(f"Protocol B\t{r50_b * 100:.2f}%\t{r100_b * 100:.2f}%\t{mr50_b * 100:.2f}%\t{mr100_b * 100:.2f}%\t{det_f1_50 * 100:.2f}%\t{det_map50 * 100:.2f}%\t{speed_str}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
