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
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fast-context-scene-graph-generation/PSG_dataset"))

from sgg_benchmark.data.datasets.data import RelationDataset
from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
)
from ckn.CKN import FC_Net
from experiments.REACT.test_single_image_inference import letterbox

LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/FastContextualSGG")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/FastContextualSGG")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("FastContextualPSGBenchmark")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

fh = logging.FileHandler(os.path.join(LOG_DIR, "psg_run.log"), mode="a", encoding="utf-8")
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def get_peak_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def undo_letterbox(boxes: np.ndarray, gain: float, pad_w: float, pad_h: float, orig_w: int, orig_h: int) -> np.ndarray:
    """Rescales letterbox-padded bounding boxes back to original image dimensions."""
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_w) / gain
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_h) / gain
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, orig_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, orig_h)
    return b


class FastContextualPSGAdapter:
    """
    Fast Contextual SGG Adapter for Panoptic Scene Graph (PSG) benchmark.
    Combines the native YOLO12m detector with the Context Knowledge Network (CKN) FC_Net('PSG').
    """

    def __init__(
        self,
        detector_path: str = "checkpoints/PSG/react++_yolo12m/yolo12m/react_pp_yolo12m.onnx",
        ckn_path: str = "model_repos/fast-context-scene-graph-generation/PSG_dataset/ckn/2619.pt",
        word_vec_path: str = ".cache/FastContextualSGG/psg_word_vec_list.pt",
        device: str = "cpu",
        conf_thres: float = 0.01,
        iou_thres: float = 0.45,
        obj_count_limit: int = 80,
    ):
        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.obj_count_limit = obj_count_limit

        # 1. Load Object Detector ONNX session
        full_det_path = os.path.join(WORKSPACE_ROOT, detector_path)
        logger.info(f"Loading ONNX detector from {full_det_path}...")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.det_session = ort.InferenceSession(full_det_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.det_session.get_inputs()[0].name

        # 2. Load CKN Model
        full_ckn_path = os.path.join(WORKSPACE_ROOT, ckn_path)
        logger.info(f"Loading CKN model from {full_ckn_path}...")
        self.ckn_model = FC_Net("PSG")
        ckn_chk = torch.load(full_ckn_path, map_location=self.device, weights_only=False)
        m_dict = ckn_chk["model"] if "model" in ckn_chk else ckn_chk
        self.ckn_model.load_state_dict(m_dict)
        self.ckn_model.to(self.device).eval()

        # 3. Load GloVe word vectors
        full_w_path = os.path.join(WORKSPACE_ROOT, word_vec_path)
        logger.info(f"Loading GloVe vectors from {full_w_path}...")
        self.word_vec_list = torch.load(full_w_path, map_location=self.device, weights_only=False)

    def predict_image(
        self,
        img_path: str,
        top_k_rels: int = 100,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        t0_e2e = time.perf_counter()

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        orig_h, orig_w = bgr.shape[:2]

        # Stage 1: Detector inference
        padded, gain, pad_w, pad_h = letterbox(bgr, 640)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

        t0_det = time.perf_counter()
        outputs = self.det_session.run(None, {self.input_name: tensor})
        t1_det = time.perf_counter()
        t_detector_ms = (t1_det - t0_det) * 1000.0

        boxes_raw = outputs[0].copy() if len(outputs) > 0 else np.zeros((0, 6), dtype=np.float32)

        # Unletterbox detected boxes to original coordinates
        if len(boxes_raw) > 0:
            boxes_raw[:, :4] = undo_letterbox(boxes_raw[:, :4], gain, pad_w, pad_h, orig_w, orig_h)
            # Filter by score threshold
            keep_mask = boxes_raw[:, 5] >= self.conf_thres
            boxes_raw = boxes_raw[keep_mask]
            # Sort by score descending and limit to top obj_count_limit
            if len(boxes_raw) > 0:
                order = np.argsort(boxes_raw[:, 5])[::-1]
                boxes_raw = boxes_raw[order[:self.obj_count_limit]]

        num_det = len(boxes_raw)

        if num_det < 2:
            t1_e2e = time.perf_counter()
            boxes = torch.from_numpy(boxes_raw[:, :4].astype(np.float32)) if num_det > 0 else torch.zeros((0, 4))
            scores = torch.from_numpy(boxes_raw[:, 5].astype(np.float32)) if num_det > 0 else torch.zeros(0)
            labels = torch.from_numpy(boxes_raw[:, 4].astype(np.int64)) if num_det > 0 else torch.zeros(0, dtype=torch.long)
            return {
                "orig_size": (orig_h, orig_w),
                "boxes": boxes,
                "pred_scores": scores,
                "pred_labels": labels,
                "rel_pair_idxs": torch.zeros((0, 2), dtype=torch.long),
                "pred_rel_scores": torch.zeros((0, 57), dtype=torch.float32),
                "pred_rel_labels": torch.zeros(0, dtype=torch.long),
            }, {
                "t_detector_ms": t_detector_ms,
                "t_contextual_ms": 0.0,
                "t_e2e_ms": (t1_e2e - t0_e2e) * 1000.0,
                "num_candidate_pairs": 0,
            }

        boxes = torch.from_numpy(boxes_raw[:, :4].astype(np.float32)).to(self.device)
        obj_scores = torch.from_numpy(boxes_raw[:, 5].astype(np.float32)).to(self.device)
        obj_labels = torch.from_numpy(boxes_raw[:, 4].astype(np.int64)).to(self.device)

        # Stage 2: Spatial & Semantic Context Feature Construction
        norm_x1 = boxes[:, 0] / orig_w
        norm_y1 = boxes[:, 1] / orig_h
        norm_x2 = boxes[:, 2] / orig_w
        norm_y2 = boxes[:, 3] / orig_h
        cx = (norm_x2 - norm_x1) / 2.0
        cy = (norm_y2 - norm_y1) / 2.0
        loc_feat = torch.stack([cx, cy, norm_x1, norm_y1, norm_x2, norm_y2], dim=1)
        word_feat = self.word_vec_list[obj_labels]
        obj = torch.cat([word_feat, loc_feat.repeat(1, 5)], dim=1)

        obj_id = torch.arange(num_det, device=self.device)
        a_id = obj_id.repeat(num_det, 1).view(-1)
        b_id = torch.repeat_interleave(obj_id, num_det, dim=0)
        obj_pair = torch.cat([a_id.unsqueeze(1), b_id.unsqueeze(1)], dim=1)
        obj_pair = obj_pair[obj_pair[:, 0] != obj_pair[:, 1]]

        rel_feature = torch.cat([
            obj[obj_pair[:, 0]],
            obj[obj_pair[:, 1]],
            obj[obj_pair[:, 1]][:, 50:] - obj[obj_pair[:, 0]][:, 50:]
        ], dim=1)

        total_pairs = len(obj_pair)

        # Stage 3: CKN Contextual Scoring
        t0_ctx = time.perf_counter()
        with torch.no_grad():
            y_ckn = self.ckn_model(rel_feature)
            pred_ckn = torch.sigmoid(y_ckn)
            ckn_conf = pred_ckn[:, 0:1]
            ckn_scores = pred_ckn[:, 1:] * ckn_conf  # 56 native predicates
            pred_ckn_full = torch.cat([torch.zeros(total_pairs, 1, device=self.device), ckn_scores], dim=1)
            rel_max_ckn, rel_argmax_ckn = torch.max(pred_ckn_full, dim=1)
        t1_ctx = time.perf_counter()
        t_contextual_ms = (t1_ctx - t0_ctx) * 1000.0

        # Stage 4: SGDet Triplet Score Calculation & Top-K Ranking
        pair_conf = obj_scores[obj_pair[:, 0]] * obj_scores[obj_pair[:, 1]] * rel_max_ckn
        sort_order = torch.argsort(pair_conf, descending=True)
        if top_k_rels is not None and len(sort_order) > top_k_rels:
            sort_order = sort_order[:top_k_rels]

        final_pairs = obj_pair[sort_order].cpu()
        final_scores = pred_ckn_full[sort_order].cpu()
        final_labels = rel_argmax_ckn[sort_order].cpu()

        t1_e2e = time.perf_counter()
        t_e2e_ms = (t1_e2e - t0_e2e) * 1000.0

        return {
            "orig_size": (orig_h, orig_w),
            "boxes": boxes.cpu(),
            "pred_scores": obj_scores.cpu(),
            "pred_labels": obj_labels.cpu(),
            "rel_pair_idxs": final_pairs,
            "pred_rel_scores": final_scores,
            "pred_rel_labels": final_labels,
        }, {
            "t_detector_ms": t_detector_ms,
            "t_contextual_ms": t_contextual_ms,
            "t_e2e_ms": t_e2e_ms,
            "num_candidate_pairs": total_pairs,
        }


def run_psg_inference(
    adapter: FastContextualPSGAdapter,
    dataset: RelationDataset,
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    num_images = min(max_images, len(dataset))
    logger.info(f"Running Fast Contextual SGG inference across {num_images} PSG images...")

    predictions = []
    det_times = []
    ctx_times = []
    e2e_times = []
    cand_pair_counts = []

    for idx in range(num_images):
        img_info = dataset.get_img_info(idx)
        img_id = img_info.get("image_id", img_info.get("id", idx))
        file_name = img_info.get("file_name", f"{img_id}.jpg")
        img_path = os.path.join(dataset.img_dir, file_name)

        pred_dict, timing = adapter.predict_image(img_path, top_k_rels=top_k_rels)
        pred_dict["image_id"] = img_id
        pred_dict["dataset_idx"] = idx

        predictions.append(pred_dict)
        det_times.append(timing["t_detector_ms"])
        ctx_times.append(timing["t_contextual_ms"])
        e2e_times.append(timing["t_e2e_ms"])
        cand_pair_counts.append(timing["num_candidate_pairs"])

        if (idx + 1) % 25 == 0 or (idx + 1) == num_images:
            avg_e2e = np.mean(e2e_times)
            avg_det = np.mean(det_times)
            avg_ctx = np.mean(ctx_times)
            peak_ram = get_peak_memory_mb()
            logger.info(
                f"Processed {idx + 1}/{num_images} images | "
                f"E2E: {avg_e2e:.1f} ms (Det: {avg_det:.1f} ms, Ctx: {avg_ctx:.1f} ms) | "
                f"RAM: {peak_ram:.1f} MB"
            )

    mean_det = float(np.mean(det_times))
    mean_ctx = float(np.mean(ctx_times))
    mean_e2e = float(np.mean(e2e_times))
    fps = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0
    peak_mem = get_peak_memory_mb()

    profile = {
        "mean_detector_latency_ms": mean_det,
        "mean_contextual_latency_ms": mean_ctx,
        "mean_e2e_latency_ms": mean_e2e,
        "fps": fps,
        "peak_rss_mb": peak_mem,
        "mean_candidate_pairs": float(np.mean(cand_pair_counts)),
    }

    if cache_path:
        torch.save({"predictions": predictions, "profile": profile}, cache_path)
        logger.info(f"Saved predictions and profile to {cache_path}")

    return predictions, profile


def evaluate_psg_recalls(
    predictions: List[Dict[str, Any]],
    dataset: RelationDataset,
    k_list: List[int] = [50, 100]
) -> Dict[str, Any]:
    """
    Evaluates Native PSG Scene Graph Detection (SGDet) Recall and Mean Recall.
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


def evaluate_detection_mAP(predictions: List[Dict[str, Any]], dataset: RelationDataset) -> Tuple[float, float]:
    """
    Evaluates COCO mAP@50 and calculates F1 score at IoU=0.5.
    """
    logger.info("Computing detection mAP@50 and F1@50 on PSG...")
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
    parser = argparse.ArgumentParser(description="Fast Contextual SGG Benchmark on PSG Dataset")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    parser.add_argument("--ckn_weights", type=str, default="model_repos/fast-context-scene-graph-generation/PSG_dataset/ckn/2619.pt", help="CKN checkpoint path")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Starting Fast Contextual SGG Benchmark Pipeline on PSG Dataset")
    logger.info(f"Max Images: {args.max_images} | Device: {args.device} | CKN: {args.ckn_weights}")
    logger.info("=" * 80)

    annotation_file = os.path.join(WORKSPACE_ROOT, "evaluation_sets/PSG_200_annotations.coco.json")
    img_dir = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/PSG")

    psg_dataset = RelationDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        filter_empty_rels=False,
        filter_duplicate_rels=False,
        filter_non_overlap=False
    )
    logger.info(f"Loaded PSG evaluation dataset: {len(psg_dataset)} images")

    adapter = FastContextualPSGAdapter(
        ckn_path=args.ckn_weights,
        device=args.device
    )

    mode_str = "smoke" if args.max_images <= 5 else "full"
    cache_file = os.path.join(CACHE_DIR, f"raw_preds_PSG_csgg_{mode_str}.pt")

    predictions, profile = run_psg_inference(
        adapter=adapter,
        dataset=psg_dataset,
        max_images=args.max_images,
        cache_path=cache_file
    )

    # Detection metrics
    map_50, f1_50 = evaluate_detection_mAP(predictions, psg_dataset)
    logger.info(f"Object Detection on PSG: mAP@50 = {map_50:.4f} | F1@50 = {f1_50:.4f}")

    # SGG Recall metrics
    sgg_metrics = evaluate_psg_recalls(predictions, psg_dataset, k_list=[50, 100])

    r50 = sgg_metrics["recalls_eligible"]["R@50"]
    r100 = sgg_metrics["recalls_eligible"]["R@100"]
    mr50 = sgg_metrics["mean_recalls"]["mR@50"]
    mr100 = sgg_metrics["mean_recalls"]["mR@100"]
    latency_ms = profile["mean_e2e_latency_ms"]
    fps = profile["fps"]

    speed_str = f"{latency_ms:.1f} ms ({fps:.2f} FPS)"

    logger.info("=" * 80)
    logger.info("FINAL BENCHMARK SCORES FOR FAST CONTEXTUAL SGG ON PSG:")
    logger.info(f"R@50:            {r50*100:.2f}%")
    logger.info(f"R@100:           {r100*100:.2f}%")
    logger.info(f"mR@50:           {mr50*100:.2f}%")
    logger.info(f"mR@100:          {mr100*100:.2f}%")
    logger.info(f"F1@50:           {f1_50*100:.2f}% (or {f1_50:.4f})")
    logger.info(f"Detection mAP@50: {map_50*100:.2f}% (or {map_50:.4f})")
    logger.info(f"Speed:           {speed_str}")
    logger.info("=" * 80)

    # Save metrics JSON & CSV
    out_dict = {
        "model": "Fast Contextual SGG (C-SGG)",
        "dataset": "PSG (Panoptic Scene Graph)",
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

    metrics_json_path = os.path.join(LOG_DIR, "psg_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    logger.info(f"Saved PSG metrics to {metrics_json_path}")

    metrics_csv_path = os.path.join(LOG_DIR, "psg_metrics.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "R@50", "R@100", "mR@50", "mR@100", "F1@50", "Detection mAP@50", "Speed"])
        writer.writerow([
            "Fast Contextual SGG",
            "PSG",
            f"{r50*100:.2f}%",
            f"{r100*100:.2f}%",
            f"{mr50*100:.2f}%",
            f"{mr100*100:.2f}%",
            f"{f1_50*100:.2f}%",
            f"{map_50*100:.2f}%",
            speed_str
        ])
    logger.info(f"Saved PSG metrics CSV to {metrics_csv_path}")

    # Print requested table format
    print("\n" + "=" * 80)
    print("REQUESTED BENCHMARK OUTPUT TABLE:")
    print("R@50\tR@100\tmR@50\tmR@100\tF1@50\tDetection mAP@50\tSpeed")
    print(f"{r50*100:.2f}%\t{r100*100:.2f}%\t{mr50*100:.2f}%\t{mr100*100:.2f}%\t{f1_50*100:.2f}%\t{map_50*100:.2f}%\t{speed_str}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
