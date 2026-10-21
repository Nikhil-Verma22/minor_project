import os
import sys
import time
import json
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
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

LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/REACT")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/REACT")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("REACTIndoorVGBenchmark")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(os.path.join(LOG_DIR, "indoorvg_run.log"), mode="a", encoding="utf-8")
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


def load_cached_or_run_inference(
    onnx_path: str,
    dataset: RelationDataset,
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached REACT IndoorVG predictions from {cache_path}...")
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
    predictions = []
    fwd_times = []
    e2e_times = []
    peak_mem = 0.0

    logger.info(f"Executing fresh REACT inference on IndoorVG ({num_images} images)...")

    for idx in range(num_images):
        t0_e2e = time.perf_counter()
        img_info = dataset.get_img_info(idx)
        img_id = img_info.get("image_id", idx)
        img_path = dataset.filenames[idx]

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        orig_h, orig_w = bgr.shape[:2]

        padded, gain, pad_w, pad_h = letterbox(bgr, 640)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

        t0_fwd = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        t1_fwd = time.perf_counter()

        boxes_raw = outputs[0].copy() if len(outputs) > 0 else np.zeros((0, 6), dtype=np.float32)
        rels_raw = outputs[1].copy() if len(outputs) > 1 else np.zeros((0, 5), dtype=np.float32)

        # Rescale boxes back to original image coordinates
        if len(boxes_raw) > 0:
            boxes_raw = undo_letterbox(boxes_raw, gain, pad_w, pad_h, orig_w, orig_h)

        # Sort and filter top-K relations by triplet score (col 3)
        if len(rels_raw) > top_k_rels:
            order = np.argsort(rels_raw[:, 3])[::-1][:top_k_rels]
            rels_raw = rels_raw[order]

        boxes = torch.from_numpy(boxes_raw[:, :4].astype(np.float32)) if len(boxes_raw) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        pred_labels = torch.from_numpy(boxes_raw[:, 4].astype(np.int64)) if len(boxes_raw) > 0 else torch.zeros(0, dtype=torch.int64)
        pred_scores = torch.from_numpy(boxes_raw[:, 5].astype(np.float32)) if len(boxes_raw) > 0 else torch.zeros(0, dtype=torch.float32)

        num_rel_classes = 38  # 1 background + 37 IndoorVG predicates
        if len(rels_raw) > 0:
            subj_idx = rels_raw[:, 0].astype(np.int64)
            obj_idx = rels_raw[:, 1].astype(np.int64)
            rel_label = rels_raw[:, 2].astype(np.int64)
            tri_score = rels_raw[:, 3].astype(np.float32)
            rel_score = rels_raw[:, 4].astype(np.float32)

            rel_pair_idxs = torch.from_numpy(np.stack([subj_idx, obj_idx], axis=1))

            pred_rel_scores = torch.zeros((len(rels_raw), num_rel_classes), dtype=torch.float32)
            pred_triplet_scores = torch.zeros((len(rels_raw), num_rel_classes), dtype=torch.float32)

            for i, (lbl, r_sc, t_sc) in enumerate(zip(rel_label, rel_score, tri_score)):
                if 1 <= lbl < num_rel_classes:
                    pred_rel_scores[i, lbl] = float(r_sc)
                    pred_triplet_scores[i, lbl] = float(t_sc)

            pred_rel_labels = torch.from_numpy(rel_label)
        else:
            rel_pair_idxs = torch.zeros((0, 2), dtype=torch.int64)
            pred_rel_scores = torch.zeros((0, num_rel_classes), dtype=torch.float32)
            pred_triplet_scores = torch.zeros((0, num_rel_classes), dtype=torch.float32)
            pred_rel_labels = torch.zeros(0, dtype=torch.int64)

        pred_dict = {
            "image_id": img_id,
            "orig_size": (orig_h, orig_w),
            "boxes": boxes,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
            "rel_pair_idxs": rel_pair_idxs,
            "pred_rel_scores": pred_rel_scores,
            "pred_triplet_scores": pred_triplet_scores,
            "pred_rel_labels": pred_rel_labels,
        }
        predictions.append(pred_dict)
        t1_e2e = time.perf_counter()

        fwd_times.append((t1_fwd - t0_fwd) * 1000.0)
        e2e_times.append((t1_e2e - t0_e2e) * 1000.0)

        current_mem = get_peak_memory_mb()
        if current_mem > peak_mem:
            peak_mem = current_mem

        if (idx + 1) % 25 == 0 or idx == num_images - 1:
            logger.info(
                f"Progress: [{idx + 1:3d}/{num_images:3d}] | "
                f"Fwd: {np.mean(fwd_times):.1f} ms | "
                f"E2E: {np.mean(e2e_times):.1f} ms | "
                f"RAM: {current_mem:.1f} MB"
            )

    mean_fwd = float(np.mean(fwd_times))
    std_fwd = float(np.std(fwd_times))
    mean_e2e = float(np.mean(e2e_times))
    std_e2e = float(np.std(e2e_times))
    fps = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0

    profile = {
        "num_images": num_images,
        "mean_forward_latency_ms": mean_fwd,
        "std_forward_latency_ms": std_fwd,
        "mean_e2e_latency_ms": mean_e2e,
        "std_e2e_latency_ms": std_e2e,
        "fps": fps,
        "peak_rss_mb": peak_mem,
    }

    if cache_path:
        torch.save({"predictions": predictions, "profile": profile}, cache_path)
        logger.info(f"Saved inference predictions to cache: {cache_path}")

    return predictions, profile


def evaluate_detection_mAP(predictions: List[Dict[str, Any]], dataset: RelationDataset) -> Tuple[float, float]:
    """Computes COCO mAP@50 and F1 score at IoU=0.5."""
    logger.info("Computing detection mAP@50 and F1@50 on IndoorVG...")
    images, anns = [], []
    ann_id = 0

    for idx, pred in enumerate(predictions):
        img_info = dataset.get_img_info(idx)
        images.append({"id": idx, "width": int(img_info["width"]), "height": int(img_info["height"])})
        gt = dataset.get_groundtruth(idx, evaluation=True)
        for b, l in zip(gt["boxes"].numpy(), gt["labels"].numpy()):
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

    cats = [{"id": i, "name": n} for i, n in enumerate(dataset.ind_to_classes) if n != "__background__"]
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": anns, "categories": cats}
    coco_gt.createIndex()

    dt_anns = []
    for idx, pred in enumerate(predictions):
        for b, s, l in zip(pred["boxes"].numpy(), pred["pred_scores"].numpy(), pred["pred_labels"].numpy()):
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
    k_list: List[int] = [50, 100],
    ranking_key: str = "pred_triplet_scores"
) -> Dict[str, Any]:
    """Computes SGDet Recall and Mean Recall."""
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
        rel_scores_np = pred_dict[ranking_key].numpy()

        if len(raw_pair_idxs) == 0 or len(pred_boxes) == 0:
            for k in k_list:
                per_img_recalls_eligible[k].append(0.0)
                per_img_recalls_all[k].append(0.0)
            continue

        pred_predicate_scores = rel_scores_np[:, 1:].max(axis=1) if rel_scores_np.shape[1] > 1 else np.zeros(len(rel_scores_np))
        pred_predicate_labels = 1 + rel_scores_np[:, 1:].argmax(axis=1) if rel_scores_np.shape[1] > 1 else np.ones(len(rel_scores_np), dtype=int)

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
    }

    for k in k_list:
        results["recalls_eligible"][f"R@{k}"] = float(np.mean(per_img_recalls_eligible[k])) if per_img_recalls_eligible[k] else 0.0
        results["recalls_all_200"][f"R@{k}_all200"] = float(np.mean(per_img_recalls_all[k])) if per_img_recalls_all[k] else 0.0

        valid_m_recalls = []
        for p_cls in range(1, num_predicates):
            if class_recall_collector[k][p_cls]:
                valid_m_recalls.append(np.mean(class_recall_collector[k][p_cls]))
        results["mean_recalls"][f"mR@{k}"] = float(np.mean(valid_m_recalls)) if valid_m_recalls else 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description="REACT Benchmark on IndoorVG")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images")
    parser.add_argument("--onnx_path", type=str, default="checkpoints/REACT/react_indoorvg_yolov8m.onnx", help="ONNX checkpoint")
    args = parser.parse_args()

    annotation_file = os.path.join(WORKSPACE_ROOT, "evaluation_sets/IndoorVG_200_annotations.coco.json")
    img_dir = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/IndoorVG")

    dataset = RelationDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        filter_empty_rels=False,
        filter_duplicate_rels=False,
        filter_non_overlap=False
    )
    logger.info(f"Loaded IndoorVG evaluation dataset: {len(dataset)} images")

    cache_path = os.path.join(CACHE_DIR, "raw_preds_indoorvg_react.pt")
    predictions, profile = load_cached_or_run_inference(
        onnx_path=args.onnx_path,
        dataset=dataset,
        max_images=args.max_images,
        cache_path=cache_path
    )

    # Compute detection metrics
    map_50, f1_50 = evaluate_detection_mAP(predictions, dataset)
    logger.info(f"REACT Object Detection on IndoorVG: mAP@50 = {map_50:.4f} ({map_50*100:.2f}%) | F1@50 = {f1_50:.4f} ({f1_50*100:.2f}%)")

    # Compute SGG recall metrics with triplet ranking (author official setting)
    sgg_metrics = evaluate_sgg_recalls(predictions, dataset, k_list=[50, 100], ranking_key="pred_triplet_scores")
    r50 = sgg_metrics["recalls_eligible"]["R@50"]
    r100 = sgg_metrics["recalls_eligible"]["R@100"]
    mr50 = sgg_metrics["mean_recalls"]["mR@50"]
    mr100 = sgg_metrics["mean_recalls"]["mR@100"]
    latency_ms = profile["mean_e2e_latency_ms"]
    fps = profile["fps"]
    speed_str = f"{latency_ms:.1f} ms ({fps:.2f} FPS)"

    # Also compute SGG triplet F1 score: harmonic mean of R@50 and mR@50
    triplet_f1_50 = (2.0 * r50 * mr50 / (r50 + mr50 + 1e-8)) if (r50 + mr50) > 0 else 0.0

    logger.info("=" * 80)
    logger.info("FINAL BENCHMARK SCORES FOR REACT ON INDOORVG:")
    logger.info(f"R@50:             {r50*100:.2f}%")
    logger.info(f"R@100:            {r100*100:.2f}%")
    logger.info(f"mR@50:            {mr50*100:.2f}%")
    logger.info(f"mR@100:           {mr100*100:.2f}%")
    logger.info(f"F1@50 (Detection):{f1_50*100:.2f}% (or {f1_50:.4f})")
    logger.info(f"F1@50 (Triplet):  {triplet_f1_50*100:.2f}% (or {triplet_f1_50:.4f})")
    logger.info(f"Detection mAP@50: {map_50*100:.2f}% (or {map_50:.4f})")
    logger.info(f"Speed:            {speed_str}")
    logger.info("=" * 80)

    # Save summary JSON
    summary_results = {
        "model": "REACT (YOLOv8m)",
        "dataset": "IndoorVG",
        "num_images": len(predictions),
        "scores": {
            "R@50": r50,
            "R@100": r100,
            "mR@50": mr50,
            "mR@100": mr100,
            "F1@50": f1_50,
            "Triplet_F1@50": triplet_f1_50,
            "Detection_mAP@50": map_50,
            "Speed_latency_ms": latency_ms,
            "Speed_fps": fps,
            "Speed": speed_str,
        },
        "profile": profile,
        "sgg_metrics": sgg_metrics,
    }

    json_path = os.path.join(LOG_DIR, "indoorvg_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    logger.info(f"Saved benchmark summary JSON to {json_path}")

    # Save summary CSV
    csv_path = os.path.join(LOG_DIR, "indoorvg_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "R@50", "R@100", "mR@50", "mR@100", "F1@50", "Detection mAP@50", "Speed"])
        writer.writerow([
            "REACT (YOLOv8m)",
            "IndoorVG",
            f"{r50*100:.2f}%",
            f"{r100*100:.2f}%",
            f"{mr50*100:.2f}%",
            f"{mr100*100:.2f}%",
            f"{f1_50*100:.2f}%",
            f"{map_50*100:.2f}%",
            speed_str,
        ])
    logger.info(f"Saved benchmark CSV to {csv_path}")

    # Output exact tab-separated format requested by user
    print("\n" + "=" * 90)
    print("USER REQUESTED FORMAT OUTPUT:")
    print("R@50\tR@100\tmR@50\tmR@100\tF1@50\tDetection mAP@50\tSpeed")
    print(f"{r50*100:.2f}%\t{r100*100:.2f}%\t{mr50*100:.2f}%\t{mr100*100:.2f}%\t{f1_50*100:.2f}%\t{map_50*100:.2f}%\t{speed_str}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
