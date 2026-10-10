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

from sgg_benchmark.data.datasets.data import RelationDataset
from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
)
from experiments.REACT.ontology_projection import OntologyProjector
from experiments.FastContextualSGG.fast_contextual_adapter import FastContextualAdapter

LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/FastContextualSGG")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/FastContextualSGG")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("FastContextualBenchmark")
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


def run_inference_on_dataset(
    adapter: FastContextualAdapter,
    dataset: RelationDataset,
    variant_name: str,
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Executes model inference sequentially over the evaluation dataset.
    """
    num_images = min(max_images, len(dataset))
    logger.info(f"Running {variant_name} inference across {num_images} images...")

    predictions = []
    det_times = []
    ctx_times = []
    vis_times = []
    e2e_times = []
    cand_pair_counts = []
    ver_pair_counts = []

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
        vis_times.append(timing["t_visual_ms"])
        e2e_times.append(timing["t_e2e_ms"])
        cand_pair_counts.append(timing["num_candidate_pairs"])
        ver_pair_counts.append(timing["num_verified_pairs"])

        if (idx + 1) % 25 == 0 or (idx + 1) == num_images:
            avg_e2e = np.mean(e2e_times)
            avg_ctx = np.mean(ctx_times)
            avg_vis = np.mean(vis_times)
            peak_ram = get_peak_memory_mb()
            logger.info(
                f"[{variant_name}] Processed {idx + 1}/{num_images} images | "
                f"E2E: {avg_e2e:.1f} ms (Det: {np.mean(det_times):.1f} ms, Ctx: {avg_ctx:.1f} ms, Vis: {avg_vis:.1f} ms) | "
                f"RAM: {peak_ram:.1f} MB"
            )

    mean_det = float(np.mean(det_times))
    mean_ctx = float(np.mean(ctx_times))
    mean_vis = float(np.mean(vis_times))
    mean_e2e = float(np.mean(e2e_times))
    fps = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0
    peak_mem = get_peak_memory_mb()

    profile = {
        "variant": variant_name,
        "mean_detector_latency_ms": mean_det,
        "mean_contextual_latency_ms": mean_ctx,
        "mean_visual_latency_ms": mean_vis,
        "mean_e2e_latency_ms": mean_e2e,
        "fps": fps,
        "peak_rss_mb": peak_mem,
        "mean_candidate_pairs": float(np.mean(cand_pair_counts)),
        "mean_verified_pairs": float(np.mean(ver_pair_counts)),
    }

    if cache_path:
        torch.save({"predictions": predictions, "profile": profile}, cache_path)
        logger.info(f"Saved {variant_name} predictions and profile to {cache_path}")

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
        "view_name": view_name,
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
    logger.info("Computing detection mAP@50...")
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
    parser = argparse.ArgumentParser(description="Fast Contextual SGG Benchmark (C-SGG & CV-SGG)")
    parser.add_argument("--variants", nargs="+", default=["C-SGG", "CV-SGG"], help="Variants to evaluate")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    parser.add_argument("--cv_filter_k", type=int, default=150, help="Candidate filter K for CV-SGG")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Starting Fast Contextual SGG Benchmark Pipeline")
    logger.info(f"Variants: {args.variants} | Max Images: {args.max_images} | Device: {args.device}")
    logger.info("=" * 80)

    views = [
        ("strict", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_strict.coco.json")),
        ("extended", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_extended.coco.json")),
        ("tri_dataset", os.path.join(WORKSPACE_ROOT, "evaluation_sets/normalized/VG150_200_tri_dataset.coco.json")),
        ("native", os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")),
    ]
    img_dir = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/VG150")
    projector = OntologyProjector(os.path.join(WORKSPACE_ROOT, "evaluation_sets/evaluation_ontology.json"))

    native_dataset = RelationDataset(
        annotation_file=views[-1][1],
        img_dir=img_dir,
        filter_empty_rels=False,
        filter_duplicate_rels=False,
        filter_non_overlap=False
    )
    logger.info(f"Loaded Visual Genome evaluation dataset: {len(native_dataset)} images")

    all_benchmark_results = {}
    csv_rows = []

    for variant in args.variants:
        variant = variant.upper()
        logger.info("-" * 80)
        logger.info(f"BEGINNING BENCHMARK FOR VARIANT: {variant}")
        logger.info("-" * 80)

        adapter = FastContextualAdapter(
            variant=variant,
            device=args.device,
            cv_filter_k=args.cv_filter_k
        )

        mode_str = "smoke" if args.max_images <= 5 else "full"
        cache_file = os.path.join(CACHE_DIR, f"raw_preds_VG150_{variant.lower()}_{mode_str}.pt")

        predictions, profile = run_inference_on_dataset(
            adapter=adapter,
            dataset=native_dataset,
            variant_name=variant,
            max_images=args.max_images,
            cache_path=cache_file
        )

        map_50, f1_50 = evaluate_detection_mAP(predictions, native_dataset)
        logger.info(f"[{variant}] Object Detection: mAP@50 = {map_50:.4f} | F1@50 = {f1_50:.4f}")

        variant_results = {
            "variant": variant,
            "profile": profile,
            "detection": {
                "mAP_50": map_50,
                "F1_50": f1_50,
            },
            "views": {}
        }

        for view_name, ann_path in views:
            logger.info(f"[{variant}] Evaluating on ontology view: {view_name}")
            view_dataset = RelationDataset(
                annotation_file=ann_path,
                img_dir=img_dir,
                filter_empty_rels=False,
                filter_duplicate_rels=False,
                filter_non_overlap=False
            )
            view_res = evaluate_predictions_on_view(
                predictions=predictions,
                dataset=view_dataset,
                source_dataset_name="VG150",
                view_name=view_name,
                projector=projector
            )
            variant_results["views"][view_name] = view_res

            logger.info(
                f"[{variant} | {view_name}] R@20: {view_res['recalls_eligible']['R@20']:.4f} | "
                f"R@50: {view_res['recalls_eligible']['R@50']:.4f} | "
                f"R@100: {view_res['recalls_eligible']['R@100']:.4f} | "
                f"mR@20: {view_res['mean_recalls']['mR@20']:.4f} | "
                f"mR@50: {view_res['mean_recalls']['mR@50']:.4f} | "
                f"mR@100: {view_res['mean_recalls']['mR@100']:.4f} | "
                f"R@100 (All 200): {view_res['recalls_all_200']['R@100_all200']:.4f}"
            )

            csv_rows.append({
                "variant": variant,
                "dataset": "VG150",
                "view": view_name,
                "eligible_images": view_res["eligible_images"],
                "total_images": view_res["total_images"],
                "R@20": f"{view_res['recalls_eligible']['R@20']:.4f}",
                "R@50": f"{view_res['recalls_eligible']['R@50']:.4f}",
                "R@100": f"{view_res['recalls_eligible']['R@100']:.4f}",
                "R@20_all200": f"{view_res['recalls_all_200']['R@20_all200']:.4f}",
                "R@50_all200": f"{view_res['recalls_all_200']['R@50_all200']:.4f}",
                "R@100_all200": f"{view_res['recalls_all_200']['R@100_all200']:.4f}",
                "mR@20": f"{view_res['mean_recalls']['mR@20']:.4f}",
                "mR@50": f"{view_res['mean_recalls']['mR@50']:.4f}",
                "mR@100": f"{view_res['mean_recalls']['mR@100']:.4f}",
                "mAP@50": f"{map_50:.4f}",
                "F1@50": f"{f1_50:.4f}",
                "mean_e2e_latency_ms": f"{profile['mean_e2e_latency_ms']:.2f}",
                "mean_det_latency_ms": f"{profile['mean_detector_latency_ms']:.2f}",
                "mean_ctx_latency_ms": f"{profile['mean_contextual_latency_ms']:.2f}",
                "mean_vis_latency_ms": f"{profile['mean_visual_latency_ms']:.2f}",
                "fps": f"{profile['fps']:.2f}",
                "peak_ram_mb": f"{profile['peak_rss_mb']:.1f}",
                "mean_cand_pairs": f"{profile['mean_candidate_pairs']:.1f}",
                "mean_ver_pairs": f"{profile['mean_verified_pairs']:.1f}",
            })

        all_benchmark_results[variant] = variant_results

    # Save metrics JSON & CSV
    metrics_json_path = os.path.join(LOG_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_results, f, indent=2)
    logger.info(f"Saved complete benchmark results to {metrics_json_path}")

    metrics_csv_path = os.path.join(LOG_DIR, "metrics.csv")
    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Saved metrics CSV summary to {metrics_csv_path}")

    # Save run config
    run_cfg = {
        "benchmark": "Fast Contextual SGG Benchmark",
        "variants_evaluated": args.variants,
        "max_images": args.max_images,
        "device": args.device,
        "hardware": "Intel(R) Core(TM) Ultra 5 125H (18 threads)",
        "views": [v[0] for v in views],
        "dataset": "Visual Genome (VG150_200)",
        "cv_filter_k": args.cv_filter_k,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(LOG_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    logger.info("Benchmark execution successfully complete!")


if __name__ == "__main__":
    main()
