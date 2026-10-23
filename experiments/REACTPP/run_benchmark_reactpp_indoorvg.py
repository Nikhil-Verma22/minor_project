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

LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/REACTPP")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/REACTPP")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("REACTPPIndoorVGBenchmark")
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


def build_ontology_mappings() -> Tuple[Dict[int, int], Dict[int, int], Dict[str, Any]]:
    """
    Builds defensible, documented ontology mappings between VG150 (source domain)
    and IndoorVG (target domain).

    Rules:
    1. Exact name matches between VG150 and IndoorVG categories.
    2. Strict linguistic lemmas / true synonyms only (e.g., boy/girl/man/woman -> person).
    3. Excludes non-equivalent classes (e.g. fork -> knife, bed -> couch are NOT mapped).
    4. Unmapped categories default to 0 (__background__).
    """
    vg_coco_path = os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")
    indoor_coco_path = os.path.join(WORKSPACE_ROOT, "evaluation_sets/IndoorVG_200_annotations.coco.json")
    onto_path = os.path.join(WORKSPACE_ROOT, "evaluation_sets/evaluation_ontology.json")

    with open(vg_coco_path, "r", encoding="utf-8") as f:
        vg_data = json.load(f)
    with open(indoor_coco_path, "r", encoding="utf-8") as f:
        indoor_data = json.load(f)
    with open(onto_path, "r", encoding="utf-8") as f:
        onto_data = json.load(f)

    vg_names = {c["id"]: c["name"] for c in vg_data["categories"]}
    indoor_rev = {c["name"]: c["id"] for c in indoor_data["categories"]}

    # Defensible lemmas only: person sub-demographics, footwear, bags, luggage, tv, carpet
    lemmas = {
        "boy": "person", "girl": "person", "guy": "person", "man": "person", "men": "person",
        "woman": "person", "kid": "person", "child": "person", "player": "person", "people": "person", "lady": "person",
        "boot": "shoe", "sneaker": "shoe",
        "backpack": "bag", "handbag": "bag",
        "luggage": "suitcase",
        "tv": "television",
        "rug": "carpet",
    }

    vg_to_indoor_obj: Dict[int, int] = {}
    for vid, vname in vg_names.items():
        if vname in indoor_rev:
            vg_to_indoor_obj[vid] = indoor_rev[vname]
        elif vname in lemmas and lemmas[vname] in indoor_rev:
            vg_to_indoor_obj[vid] = indoor_rev[lemmas[vname]]

    # Predicate mapping via canonical ontology
    vg_preds = onto_data["dataset_normalizations"]["VG150"]["source_to_ontology_mapping"]
    indoor_preds = onto_data["dataset_normalizations"]["IndoorVG"]["source_to_ontology_mapping"]

    vg_to_indoor_pred: Dict[int, int] = {}
    for v_name, v_info in vg_preds.items():
        if v_name in indoor_preds:
            vg_to_indoor_pred[v_info["source_id"]] = indoor_preds[v_name]["source_id"]

    indoor_canon = {}
    for i_name, i_info in indoor_preds.items():
        canon = i_info.get("normalized_canonical_predicate")
        if canon and canon not in indoor_canon:
            indoor_canon[canon] = i_info["source_id"]

    for v_name, v_info in vg_preds.items():
        vid = v_info["source_id"]
        if vid not in vg_to_indoor_pred:
            canon = v_info.get("normalized_canonical_predicate")
            if canon and canon in indoor_canon:
                vg_to_indoor_pred[vid] = indoor_canon[canon]

    coverage_stats = {
        "vg_classes_mapped": len(vg_to_indoor_obj),
        "vg_classes_total": len(vg_names),
        "vg_coverage_pct": len(vg_to_indoor_obj) / len(vg_names) * 100.0,
        "indoor_classes_covered": len(set(vg_to_indoor_obj.values())),
        "indoor_classes_total": len(indoor_rev),
        "indoor_coverage_pct": len(set(vg_to_indoor_obj.values())) / len(indoor_rev) * 100.0,
        "vg_preds_mapped": len(vg_to_indoor_pred),
        "vg_preds_total": len(vg_preds),
    }

    logger.info(
        f"Initialized defensible ontology mappings: "
        f"{coverage_stats['vg_classes_mapped']}/{coverage_stats['vg_classes_total']} VG classes mapped "
        f"({coverage_stats['vg_coverage_pct']:.1f}%), "
        f"covering {coverage_stats['indoor_classes_covered']}/{coverage_stats['indoor_classes_total']} IndoorVG classes "
        f"({coverage_stats['indoor_coverage_pct']:.1f}%). "
        f"Predicates mapped: {coverage_stats['vg_preds_mapped']}/{coverage_stats['vg_preds_total']}."
    )

    return vg_to_indoor_obj, vg_to_indoor_pred, coverage_stats


def run_reactpp_indoorvg_inference(
    onnx_path: str,
    dataset: RelationDataset,
    vg_to_indoor_obj: Dict[int, int],
    vg_to_indoor_pred: Dict[int, int],
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached REACT++ IndoorVG predictions from {cache_path}...")
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
    logger.info(f"Executing genuine REACT++ (YOLO12m ONNX) inference on IndoorVG ({num_images} images)...")

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

        if not os.path.exists(img_path):
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

        # Rescale boxes back to original coordinates
        if len(boxes_raw) > 0:
            boxes_raw = undo_letterbox(boxes_raw, gain, pad_w, pad_h, orig_w, orig_h)

        # Sort and filter top-K relations by triplet score (col 3)
        if len(rels_raw) > top_k_rels:
            order = np.argsort(rels_raw[:, 3])[::-1][:top_k_rels]
            rels_raw = rels_raw[order]

        boxes = boxes_raw[:, :4] if len(boxes_raw) > 0 else np.zeros((0, 4), dtype=np.float32)
        raw_labels = boxes_raw[:, 4].astype(np.int64) if len(boxes_raw) > 0 else np.zeros(0, dtype=np.int64)
        scores = boxes_raw[:, 5].astype(np.float32) if len(boxes_raw) > 0 else np.zeros(0, dtype=np.float32)

        # Map object labels: VG150 ID -> IndoorVG ID (0 if unmapped)
        mapped_labels = np.array([vg_to_indoor_obj.get(int(l), 0) for l in raw_labels], dtype=np.int64)

        # Build 38-column relation score matrix (col 0: background, cols 1..37: IndoorVG predicates)
        num_pairs = len(rels_raw)
        mapped_rel_scores = np.zeros((num_pairs, 38), dtype=np.float32)
        pair_idxs = np.zeros((num_pairs, 2), dtype=np.int64)

        if num_pairs > 0:
            pair_idxs[:, 0] = rels_raw[:, 0].astype(np.int64)
            pair_idxs[:, 1] = rels_raw[:, 1].astype(np.int64)
            v_preds = rels_raw[:, 2].astype(np.int64)
            tri_scores = rels_raw[:, 3].astype(np.float32)
            rel_scores = rels_raw[:, 4].astype(np.float32)

            for i, (vp, ts, rs) in enumerate(zip(v_preds, tri_scores, rel_scores)):
                ind_p = vg_to_indoor_pred.get(int(vp), 0)
                if 1 <= ind_p < 38:
                    mapped_rel_scores[i, ind_p] = float(ts)

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
        logger.info(f"Saved REACT++ IndoorVG predictions and profile to {cache_path}")

    return predictions, profile


def evaluate_detection_mAP(predictions: List[Dict[str, Any]], dataset: RelationDataset) -> Tuple[float, float]:
    """
    Computes standard object detection mAP@50 and Detection F1@50 on IndoorVG using COCOeval.
    """
    logger.info("Computing detection mAP@50 and Detection F1@50 for REACT++ on IndoorVG...")
    images = []
    anns = []
    ann_id = 1

    for idx, pred in enumerate(predictions):
        orig_h, orig_w = pred["orig_size"]
        img_id = pred["image_id"]
        images.append({"id": img_id, "height": orig_h, "width": orig_w})

        gt_dict = dataset.get_groundtruth(idx, evaluation=False)
        gt_boxes = gt_dict["boxes"].numpy()
        gt_labels = gt_dict["labels"].numpy()

        for b, l in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = b
            w, h = max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))
            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(l),
                "bbox": [float(x1), float(y1), w, h],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt = COCO()
    categories = [{"id": i, "name": n} for i, n in enumerate(dataset.ind_to_classes) if n != "__background__"]
    coco_gt.dataset = {"images": images, "annotations": anns, "categories": categories}
    coco_gt.createIndex()

    results = []
    for pred in predictions:
        img_id = pred["image_id"]
        boxes = pred["boxes"].numpy()
        scores = pred["pred_scores"].numpy()
        labels = pred["pred_labels"].numpy()

        for b, s, l in zip(boxes, scores, labels):
            if l <= 0:
                continue
            x1, y1, x2, y2 = b
            w, h = max(0.0, float(x2 - x1)), max(0.0, float(y2 - y1))
            results.append({
                "image_id": img_id,
                "category_id": int(l),
                "bbox": [float(x1), float(y1), w, h],
                "score": float(s)
            })

    if not results:
        logger.warning("No valid predicted bounding boxes found for COCO evaluation!")
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
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
    Evaluates Scene Graph Detection (SGDet) Recall and Mean Recall on IndoorVG ground truth.
    Triplets are ranked by unified triplet confidence S = s_subj * s_obj * s_rel.
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
    parser = argparse.ArgumentParser(description="REACT++ Benchmark on IndoorVG Dataset")
    parser.add_argument("--onnx_path", type=str, default="checkpoints/REACT/react_pp_vg150_yolo12m.onnx", help="Path to ONNX weights")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images to evaluate")
    args = parser.parse_args()

    onnx_path = os.path.join(WORKSPACE_ROOT, args.onnx_path) if not os.path.isabs(args.onnx_path) else args.onnx_path

    logger.info("=" * 80)
    logger.info("Starting Genuine REACT++ (YOLO12m ONNX) Benchmark Pipeline on IndoorVG Dataset")
    logger.info(f"Model: {onnx_path} | Max Images: {args.max_images}")
    logger.info("=" * 80)

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

    vg_to_indoor_obj, vg_to_indoor_pred, coverage_stats = build_ontology_mappings()

    cache_path = os.path.join(CACHE_DIR, "raw_preds_indoorvg_reactpp.pt")
    predictions, profile = run_reactpp_indoorvg_inference(
        onnx_path=onnx_path,
        dataset=dataset,
        vg_to_indoor_obj=vg_to_indoor_obj,
        vg_to_indoor_pred=vg_to_indoor_pred,
        max_images=args.max_images,
        cache_path=cache_path
    )

    # Compute detection metrics
    map_50, det_f1_50 = evaluate_detection_mAP(predictions, dataset)

    # Compute relation metrics
    sgg_metrics = evaluate_sgg_recalls(predictions, dataset, k_list=[50, 100])
    r50 = sgg_metrics["recalls_eligible"]["R@50"]
    r100 = sgg_metrics["recalls_eligible"]["R@100"]
    mr50 = sgg_metrics["mean_recalls"]["mR@50"]
    mr100 = sgg_metrics["mean_recalls"]["mR@100"]
    latency_ms = profile["mean_e2e_latency_ms"]
    fps = profile["fps"]
    speed_str = f"{latency_ms:.1f} ms ({fps:.2f} FPS)"

    # Compute Triplet F1@50: harmonic mean of R@50 and mR@50
    triplet_f1_50 = (2.0 * r50 * mr50 / (r50 + mr50 + 1e-8)) if (r50 + mr50) > 0 else 0.0

    logger.info("=" * 80)
    logger.info("FINAL GENUINE BENCHMARK SCORES FOR REACT++ ON INDOORVG:")
    logger.info(f"R@50:                  {r50*100:.2f}%")
    logger.info(f"R@100:                 {r100*100:.2f}%")
    logger.info(f"mR@50:                 {mr50*100:.2f}%")
    logger.info(f"mR@100:                {mr100*100:.2f}%")
    logger.info(f"Detection F1@50:       {det_f1_50*100:.2f}%")
    logger.info(f"Triplet F1@50:         {triplet_f1_50*100:.2f}%")
    logger.info(f"Detection mAP@50:      {map_50*100:.2f}%")
    logger.info(f"Speed:                 {speed_str}")
    logger.info("=" * 80)

    summary_results = {
        "model": "REACT++ (YOLO12m ONNX)",
        "dataset": "IndoorVG",
        "checkpoint": onnx_path,
        "num_images": len(predictions),
        "scores": {
            "R@50": r50,
            "R@100": r100,
            "mR@50": mr50,
            "mR@100": mr100,
            "Detection_F1@50": det_f1_50,
            "Triplet_F1@50": triplet_f1_50,
            "Detection_mAP@50": map_50,
            "Speed_latency_ms": latency_ms,
            "Speed_fps": fps,
            "Speed": speed_str,
        },
        "ontology_coverage": coverage_stats,
        "profile": profile,
        "sgg_metrics": sgg_metrics,
    }

    json_path = os.path.join(LOG_DIR, "indoorvg_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    logger.info(f"Saved benchmark summary JSON to {json_path}")

    csv_path = os.path.join(LOG_DIR, "indoorvg_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "R@50", "R@100", "mR@50", "mR@100", "Detection F1@50", "Triplet F1@50", "Detection mAP@50", "Speed"])
        writer.writerow([
            "REACT++ (YOLO12m ONNX)",
            "IndoorVG",
            f"{r50*100:.2f}%",
            f"{r100*100:.2f}%",
            f"{mr50*100:.2f}%",
            f"{mr100*100:.2f}%",
            f"{det_f1_50*100:.2f}%",
            f"{triplet_f1_50*100:.2f}%",
            f"{map_50*100:.2f}%",
            speed_str,
        ])
    logger.info(f"Saved benchmark CSV to {csv_path}")

    print("\n" + "=" * 90)
    print("USER REQUESTED FORMAT OUTPUT:")
    print("R@50\tR@100\tmR@50\tmR@100\tF1@50\tDetection mAP@50\tSpeed")
    print(f"{r50*100:.2f}%\t{r100*100:.2f}%\t{mr50*100:.2f}%\t{mr100*100:.2f}%\t{det_f1_50*100:.2f}%\t{map_50*100:.2f}%\t{speed_str}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
