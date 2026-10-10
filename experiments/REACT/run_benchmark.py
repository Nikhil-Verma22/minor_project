import os
import sys
import time
import json
import csv
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from functools import reduce

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import psutil
from omegaconf import OmegaConf, open_dict

# Project paths
WORKSPACE_ROOT = "d:/Minor"
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/SGG-Benchmark"))

from sgg_benchmark.config import get_cfg
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.structures.image_list import ImageList
from sgg_benchmark.data.datasets.data import RelationDataset
from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
)
from experiments.REACT.ontology_projection import OntologyProjector
from experiments.REACT.test_single_image_inference import letterbox

# Configure logging
LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/REACT")
os.makedirs(LOG_DIR, exist_ok=True)
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/REACT")
os.makedirs(CACHE_DIR, exist_ok=True)
VIS_DIR = os.path.join(LOG_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

logger = logging.getLogger("REACTBenchmark")
logger.setLevel(logging.INFO)
# Clear existing handlers
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

def build_model_for_dataset(dataset_name: str, device: str = "cpu"):
    """Builds and returns the REACT model configured for PSG or VG150."""
    logger.info(f"Building REACT model for {dataset_name} on {device}...")
    
    cfg_raw = OmegaConf.load(os.path.join(WORKSPACE_ROOT, "model_repos/SGG-Benchmark/configs/hydra/PSG/REACT.yaml"))
    
    if dataset_name.upper() == "PSG":
        ckpt_dir = os.path.join(WORKSPACE_ROOT, "checkpoints/REACT/react_PSG/react_PSG")
        ckpt_path = os.path.join(ckpt_dir, "best_model_epoch_11.pth")
        num_obj_classes = 134
        num_rel_classes = 57
    elif dataset_name.upper() == "VG150":
        ckpt_dir = os.path.join(WORKSPACE_ROOT, "checkpoints/REACT/react_VG150")
        ckpt_path = os.path.join(ckpt_dir, "best_model_epoch_9.pth")
        num_obj_classes = 151
        num_rel_classes = 51
    else:
        raise ValueError(f"Unsupported dataset for model building: {dataset_name}")

    with open_dict(cfg_raw):
        cfg_raw.model.device = device
        cfg_raw.output_dir = ckpt_dir
        cfg_raw.glove_dir = os.path.join(WORKSPACE_ROOT, ".cache/REACT/glove")
        cfg_raw.model.roi_box_head.num_classes = num_obj_classes
        cfg_raw.model.roi_box_head.mlp_head_dim = 512
        cfg_raw.model.roi_box_head.pooler_scales = [0.0625]
        cfg_raw.model.roi_relation_head.mlp_head_dim = 512
        cfg_raw.model.roi_relation_head.num_classes = num_rel_classes
        cfg_raw.datasets.name = dataset_name.upper()

    cfg = get_cfg(cfg_raw)
    model = build_detection_model(cfg)
    model.eval()

    logger.info(f"Loading weights from {ckpt_path}...")
    checkpointer = DetectronCheckpointer(cfg, model)
    checkpointer.load(ckpt_path)
    return model, cfg

def run_model_inference_on_dataset(
    model,
    dataset: RelationDataset,
    max_images: int = 200,
    cache_path: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Runs model inference over dataset images, returning raw prediction dictionaries
    and performance profiling statistics. Caches raw predictions if cache_path is specified.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached predictions from {cache_path}...")
        cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached_data["predictions"][:max_images], cached_data["profile"]

    predictions = []
    num_images = min(len(dataset), max_images)
    forward_times = []
    e2e_times = []
    peak_mem = 0.0

    logger.info(f"Running inference across {num_images} images...")
    for idx in range(num_images):
        t0_e2e = time.perf_counter()
        img_info = dataset.get_img_info(idx)
        img_id = img_info["image_id"]
        img_path = dataset.filenames[idx]
        
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        orig_h, orig_w = bgr.shape[:2]

        padded, gain, pad_w, pad_h = letterbox(bgr, 640)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        il = ImageList(tensor[None], [(orig_h, orig_w)])

        t0_fwd = time.perf_counter()
        with torch.no_grad():
            output = model(il)
        t1_fwd = time.perf_counter()

        res = output[0]
        # Keep predictions on CPU
        pred_dict = {
            "image_id": img_id,
            "orig_size": (orig_h, orig_w),
            "boxes": res["boxes"].cpu(),
            "pred_scores": res["pred_scores"].cpu(),
            "pred_labels": res["pred_labels"].cpu(),
            "rel_pair_idxs": res["rel_pair_idxs"].cpu(),
            "pred_rel_scores": res["pred_rel_scores"].cpu(),
        }
        predictions.append(pred_dict)

        t1_e2e = time.perf_counter()
        fwd_ms = (t1_fwd - t0_fwd) * 1000.0
        e2e_ms = (t1_e2e - t0_e2e) * 1000.0
        forward_times.append(fwd_ms)
        e2e_times.append(e2e_ms)

        current_mem = get_peak_memory_mb()
        if current_mem > peak_mem:
            peak_mem = current_mem

        if (idx + 1) % 25 == 0 or idx == num_images - 1:
            logger.info(f"Processed {idx + 1}/{num_images} images | Fwd: {np.mean(forward_times):.1f} ms | E2E: {np.mean(e2e_times):.1f} ms | RAM: {current_mem:.1f} MB")

    total_time_s = sum(e2e_times) / 1000.0
    fps = float(num_images / total_time_s) if total_time_s > 0 else 0.0

    profile = {
        "num_images": num_images,
        "mean_forward_latency_ms": float(np.mean(forward_times)),
        "std_forward_latency_ms": float(np.std(forward_times)),
        "mean_e2e_latency_ms": float(np.mean(e2e_times)),
        "std_e2e_latency_ms": float(np.std(e2e_times)),
        "fps": fps,
        "peak_rss_mb": peak_mem,
    }

    if cache_path:
        torch.save({"predictions": predictions, "profile": profile}, cache_path)
        logger.info(f"Saved predictions and profile to {cache_path}")

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
    Evaluates model predictions against a target ontology view ground truth dataset.
    Computes:
    - Recall@K for eligible images (len(gt_rels) > 0)
    - Recall@K for all 200 images (empty images = 0.0)
    - mean Recall@K across all valid predicate classes
    - Per-predicate recall
    - Eligible vs empty image counts
    """
    num_images = len(predictions)
    target_ind_to_predicates = dataset.ind_to_predicates
    num_predicates = len(target_ind_to_predicates)

    # Determine source native predicate vocabulary
    if source_dataset_name.upper() == "PSG":
        with open(os.path.join(WORKSPACE_ROOT, "evaluation_sets/PSG_200_annotations.coco.json")) as f:
            psg_raw = json.load(f)
        source_ind_to_predicates = ["__background__"] + [r["name"] for r in psg_raw["rel_categories"]]
    elif source_dataset_name.upper() == "VG150":
        with open(os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")) as f:
            vg_raw = json.load(f)
        source_ind_to_predicates = ["__background__"] + [r["name"] for r in vg_raw["rel_categories"]]
    else:
        source_ind_to_predicates = target_ind_to_predicates

    eligible_image_indices = []
    empty_image_indices = []

    # Containers for metrics
    per_img_recalls_eligible = {k: [] for k in k_list}
    per_img_recalls_all = {k: [] for k in k_list}
    
    # Class-level recall collector: per class, list of recalls
    class_recall_collector = {k: [[] for _ in range(num_predicates)] for k in k_list}
    class_gt_counts = [0] * num_predicates

    total_gt_relations = 0
    total_pred_relations = 0

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

        # Count ground truth per predicate
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

        # Project relation scores if needed
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
            p_recalls = class_recall_collector[k][p_cls]
            if len(p_recalls) > 0:
                mr_per_class.append(float(np.mean(p_recalls)))
        results["mean_recalls"][f"mR@{k}"] = float(np.mean(mr_per_class)) if len(mr_per_class) > 0 else 0.0

    # Per-predicate breakdown at K=100
    for p_cls in range(1, num_predicates):
        p_name = target_ind_to_predicates[p_cls]
        p_recalls = class_recall_collector[100][p_cls]
        results["predicate_gt_counts"][p_name] = class_gt_counts[p_cls]
        results["per_predicate_recall_100"][p_name] = float(np.mean(p_recalls)) if len(p_recalls) > 0 else 0.0

    return results

def evaluate_detection_mAP(predictions: List[Dict[str, Any]], dataset: RelationDataset) -> float:
    """Evaluates object detection mAP@50 using pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    anns = []
    images = []
    ann_id = 0

    for idx, pred in enumerate(predictions):
        gt = dataset.get_groundtruth(idx, evaluation=True)
        img_info = dataset.get_img_info(idx)
        images.append({"id": idx, "width": img_info["width"], "height": img_info["height"]})

        gt_boxes = gt["boxes"].numpy()
        gt_labels = gt["labels"].numpy()
        for cls, box in zip(gt_labels, gt_boxes):
            w = max(0.0, box[2] - box[0])
            h = max(0.0, box[3] - box[1])
            anns.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(cls),
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

def run_experiment_suite(smoke_test: bool = False):
    """Executes the full benchmark evaluation suite across supported combinations and views."""
    logger.info("=" * 80)
    logger.info(f"STARTING REACT BENCHMARK SUITE | Mode: {'SMOKE TEST (3 IMAGES)' if smoke_test else 'FULL EVALUATION (200 IMAGES)'}")
    logger.info("=" * 80)

    max_imgs = 3 if smoke_test else 200
    projector = OntologyProjector(os.path.join(WORKSPACE_ROOT, "evaluation_sets/evaluation_ontology.json"))

    experiments = [
        {
            "model_dataset": "PSG",
            "eval_dataset": "PSG",
            "img_dir": os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/PSG"),
            "views": [
                ("strict", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/PSG_200_strict.coco.json")),
                ("extended", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/PSG_200_extended.coco.json")),
                ("tri_dataset", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/PSG_200_tri_dataset.coco.json")),
                ("native", os.path.join(WORKSPACE_ROOT, "evaluation_sets/PSG_200_annotations.coco.json")),
            ],
            "checkpoint": "checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth",
            "raw_cache": os.path.join(CACHE_DIR, f"raw_preds_PSG_{'smoke' if smoke_test else 'full'}.pt")
        },
        {
            "model_dataset": "VG150",
            "eval_dataset": "VG150",
            "img_dir": os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/VG150"),
            "views": [
                ("strict", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_strict.coco.json")),
                ("extended", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_extended.coco.json")),
                ("tri_dataset", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_tri_dataset.coco.json")),
                ("native", os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")),
            ],
            "checkpoint": "checkpoints/REACT/react_VG150/best_model_epoch_9.pth",
            "raw_cache": os.path.join(CACHE_DIR, f"raw_preds_VG150_{'smoke' if smoke_test else 'full'}.pt")
        }
    ]

    all_metrics = []
    all_profiles = {}

    for exp in experiments:
        m_ds = exp["model_dataset"]
        e_ds = exp["eval_dataset"]
        logger.info("-" * 80)
        logger.info(f"Running Experiment: Model={m_ds} on Eval={e_ds}")
        logger.info("-" * 80)

        # Build model and run inference once
        model, cfg = build_model_for_dataset(m_ds, device="cpu")
        
        # Load native dataset for inference filenames
        native_ann = exp["views"][-1][1]
        native_dataset = RelationDataset(
            annotation_file=native_ann,
            img_dir=exp["img_dir"],
            filter_empty_rels=False,
            filter_duplicate_rels=False,
            filter_non_overlap=False
        )

        preds, profile = run_model_inference_on_dataset(
            model=model,
            dataset=native_dataset,
            max_images=max_imgs,
            cache_path=exp["raw_cache"]
        )
        all_profiles[m_ds] = profile

        # Evaluate detection mAP@50
        logger.info(f"Evaluating Object Detection mAP@50 for {m_ds} detector...")
        det_map50 = evaluate_detection_mAP(preds, native_dataset)
        logger.info(f"Detection mAP@50 ({m_ds}): {det_map50:.4f}")

        # Now evaluate across each view
        for view_name, ann_file in exp["views"]:
            logger.info(f"Evaluating Model={m_ds} on View={view_name} ({ann_file})...")
            view_dataset = RelationDataset(
                annotation_file=ann_file,
                img_dir=exp["img_dir"],
                filter_empty_rels=False,
                filter_duplicate_rels=False,
                filter_non_overlap=False
            )

            view_eval = evaluate_predictions_on_view(
                predictions=preds,
                dataset=view_dataset,
                source_dataset_name=m_ds,
                view_name=view_name,
                projector=projector,
                k_list=[20, 50, 100]
            )

            record = {
                "model": "REACT",
                "checkpoint": exp["checkpoint"],
                "source_dataset": m_ds,
                "eval_dataset": e_ds,
                "ontology_view": view_name,
                "task": "SGDet",
                "status": "COMPLETED",
                "expected_images": max_imgs,
                "processed_images": len(preds),
                "eligible_images": view_eval["eligible_images"],
                "empty_images": view_eval["empty_images"],
                "total_gt_relations": view_eval["total_gt_relations"],
                "detection_mAP50": det_map50,
                # Eligible Recall
                "R@20": view_eval["recalls_eligible"]["R@20"],
                "R@50": view_eval["recalls_eligible"]["R@50"],
                "R@100": view_eval["recalls_eligible"]["R@100"],
                # All 200 Sensitivity Recall
                "R@20_all200": view_eval["recalls_all_200"]["R@20_all200"],
                "R@50_all200": view_eval["recalls_all_200"]["R@50_all200"],
                "R@100_all200": view_eval["recalls_all_200"]["R@100_all200"],
                # Mean Recall
                "mR@20": view_eval["mean_recalls"]["mR@20"],
                "mR@50": view_eval["mean_recalls"]["mR@50"],
                "mR@100": view_eval["mean_recalls"]["mR@100"],
                # Latency & Profiling
                "mean_forward_latency_ms": profile["mean_forward_latency_ms"],
                "mean_e2e_latency_ms": profile["mean_e2e_latency_ms"],
                "fps": profile["fps"],
                "peak_rss_mb": profile["peak_rss_mb"],
                "per_predicate_recall_100": view_eval["per_predicate_recall_100"],
                "predicate_gt_counts": view_eval["predicate_gt_counts"]
            }
            all_metrics.append(record)
            logger.info(
                f"[{m_ds} | {view_name.upper()}] "
                f"R@20: {record['R@20']:.4f} | R@50: {record['R@50']:.4f} | R@100: {record['R@100']:.4f} | "
                f"mR@50: {record['mR@50']:.4f} | mR@100: {record['mR@100']:.4f} | "
                f"All-200 R@50: {record['R@50_all200']:.4f} | Eligible: {record['eligible_images']}/{max_imgs}"
            )

    # Save metrics JSON & CSV
    json_path = os.path.join(LOG_DIR, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")

    csv_path = os.path.join(LOG_DIR, "metrics.csv")
    csv_fields = [
        "model", "checkpoint", "source_dataset", "eval_dataset", "ontology_view", "task", "status",
        "expected_images", "processed_images", "eligible_images", "empty_images", "total_gt_relations",
        "detection_mAP50", "R@20", "R@50", "R@100", "R@20_all200", "R@50_all200", "R@100_all200",
        "mR@20", "mR@50", "mR@100", "mean_forward_latency_ms", "mean_e2e_latency_ms", "fps", "peak_rss_mb"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for m in all_metrics:
            row = {k: m[k] for k in csv_fields}
            writer.writerow(row)
    logger.info(f"Saved metrics CSV to {csv_path}")

    # Save run_config.json
    config_record = {
        "benchmark": "REACT Pretrained Evaluation",
        "hardware": {
            "cpu": "Intel(R) Core(TM) Ultra 5 125H",
            "logical_processors": 18,
            "device": "cpu"
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "ultralytics": "8.3.250"
        },
        "input_resolution": [640, 640],
        "batch_size": 1,
        "mode": "smoke_test" if smoke_test else "full_evaluation",
        "supported_datasets": ["PSG", "VG150"],
        "evaluated_views": ["strict", "extended", "tri_dataset", "native"],
        "metrics_evaluated": [
            "SGDet Recall@20/50/100 (Eligible Images)",
            "SGDet Recall@20/50/100 (All 200 Images Sensitivity)",
            "SGDet mean Recall@20/50/100",
            "Per-predicate Recall@100",
            "Object Detection mAP@50"
        ]
    }
    with open(os.path.join(LOG_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    logger.info("Evaluation suite finished successfully.")
    return all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run 3-image smoke test")
    args = parser.parse_args()
    run_experiment_suite(smoke_test=args.smoke_test)
