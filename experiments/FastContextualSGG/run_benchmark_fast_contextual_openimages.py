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
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fast-context-scene-graph-generation/VG_dataset"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fast-context-scene-graph-generation/OID_dataset"))

from sgg_benchmark.data.datasets.data import RelationDataset
from sgg_benchmark.data.datasets.evaluation.sgg_metrics import (
    _compute_pred_matches,
    _triplet,
)
from experiments.FastContextualSGG.fast_contextual_adapter import FastContextualAdapter
from ckn.CKN import FC_Net as FC_Net_OID

LOG_DIR = os.path.join(WORKSPACE_ROOT, "results/FastContextualSGG")
CACHE_DIR = os.path.join(WORKSPACE_ROOT, ".cache/FastContextualSGG")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("FastContextualOpenImagesBenchmark")
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


def build_ontology_mappings() -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Constructs semantic cross-dataset mappings from VG150 (Fast Contextual SGG training ontology)
    to OpenImages V6 (75 object classes + 31 relationship predicates).
    Also returns CKN_OID (30 predicates) -> OpenImages V6 (31 predicates) direct mapping.
    """
    vg_annot_path = os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")
    oid_annot_path = os.path.join(WORKSPACE_ROOT, "evaluation_sets/OpenImages_200_annotations.coco.json")
    ontology_path = os.path.join(WORKSPACE_ROOT, "evaluation_sets/evaluation_ontology.json")

    with open(vg_annot_path, "r", encoding="utf-8") as f:
        vg_data = json.load(f)
    with open(oid_annot_path, "r", encoding="utf-8") as f:
        oid_data = json.load(f)
    with open(ontology_path, "r", encoding="utf-8") as f:
        onto_data = json.load(f)

    vg_names = {c["id"]: c["name"].lower() for c in vg_data["categories"]}
    oid_cats = {c["name"].lower(): c["id"] for c in oid_data["categories"]}

    synonyms = {
        # people
        "man": "man", "men": "man", "guy": "man",
        "woman": "woman", "lady": "woman",
        "boy": "boy", "kid": "boy", "child": "boy",
        "girl": "girl",
        "person": "man", "people": "man", "player": "man",
        # vehicles
        "car": "car", "vehicle": "car",
        "bus": "bus",
        "truck": "van",
        "bike": "bicycle", "bicycle": "bicycle",
        "motorcycle": "motorcycle",
        # animals
        "dog": "dog",
        # furniture & house
        "chair": "chair", "seat": "chair", "bench": "chair",
        "table": "table", "counter": "table",
        "desk": "desk",
        "bed": "bed",
        "plant": "houseplant", "flower": "houseplant",
        "tree": "tree",
        # kitchen / dining
        "bottle": "bottle",
        "cup": "coffee cup", "mug": "mug",
        "glass": "wine glass",
        "fork": "fork",
        "plate": "platter", "bowl": "platter",
        # apparel / accessories
        "hat": "fedora", "cap": "fedora", "helmet": "bicycle helmet",
        "bag": "handbag", "handbag": "handbag", "backpack": "handbag", "suitcase": "handbag",
        "shoe": "sandal", "boot": "sandal", "sneaker": "sandal",
        # food
        "orange": "orange", "fruit": "orange", "banana": "orange", "apple": "orange",
        # sports & recreation
        "skateboard": "skateboard",
        "racket": "table tennis racket",
        # electronics & instruments
        "guitar": "guitar",
        "piano": "piano",
        "drum": "drum",
        "flute": "flute",
        "violin": "violin",
        "microphone": "microphone",
        "vase": "flowerpot", "pot": "flowerpot",
        "stand": "stand",
    }

    # 1. Object Mapping: VG150 ID (1..150) -> OpenImages ID (1..75)
    vg_to_oid_obj: Dict[int, int] = {}
    for vid, vname in vg_names.items():
        if vname in oid_cats:
            vg_to_oid_obj[vid] = oid_cats[vname]
        elif vname in synonyms and synonyms[vname] in oid_cats:
            vg_to_oid_obj[vid] = oid_cats[synonyms[vname]]

    # 2. Predicate Mapping from VG150 via ontology: VG150 Predicate ID (1..50) -> OpenImages ID (1..31)
    vg_preds = onto_data["dataset_normalizations"]["VG150"]["source_to_ontology_mapping"]
    oid_preds = onto_data["dataset_normalizations"]["OpenImagesV6"]["source_to_ontology_mapping"]

    c2o = {}
    for o_name, o_info in oid_preds.items():
        canon = o_info.get("normalized_canonical_predicate")
        if canon and canon not in c2o:
            c2o[canon] = o_info["source_id"]

    vg_to_oid_pred: Dict[int, int] = {}
    for v_name, v_info in vg_preds.items():
        if v_name in oid_preds:
            vg_to_oid_pred[v_info["source_id"]] = oid_preds[v_name]["source_id"]
        else:
            canon = v_info.get("normalized_canonical_predicate")
            if canon and canon in c2o:
                vg_to_oid_pred[v_info["source_id"]] = c2o[canon]

    # Additional manual mappings based on VG physical semantics to OpenImages:
    manual_vg_to_oid = {
        13: 8,  # covering -> on
        24: 8,  # laying on -> on
        26: 8,  # lying on -> on
        40: 8,  # sitting on -> on
        41: 8,  # standing on -> on
        50: 21, # with -> interacts_with
    }
    for vid, oid_id in manual_vg_to_oid.items():
        if vid not in vg_to_oid_pred:
            vg_to_oid_pred[vid] = oid_id

    # 3. Direct CKN_OID (1..30) -> OpenImages Ground Truth (1..31) mapping
    author_preds = [
        "__background__", "at", "holds", "wears", "surf", "hang", "drink", "holding_hands", "on", "ride", "dance",
        "skateboard", "catch", "highfive", "inside_of", "eat", "cut", "contain", "handshake", "kiss", "talk_on_phone",
        "interacts_with", "under", "hug", "throw", "hits", "snowboard", "kick", "ski", "plays", "read"
    ]
    p_names_oid = {p["id"]: p["name"] for p in oid_data["rel_categories"]}
    ckn_oid_to_gt = {}
    for a_idx, a_name in enumerate(author_preds):
        if a_idx == 0:
            continue
        for o_id, o_name in p_names_oid.items():
            if a_name == o_name:
                ckn_oid_to_gt[a_idx] = o_id
                break

    logger.info(
        f"Initialized ontology mappings: {len(vg_to_oid_obj)}/150 object classes, "
        f"{len(vg_to_oid_pred)}/50 VG predicates, {len(ckn_oid_to_gt)}/30 OID CKN predicates"
    )
    return vg_to_oid_obj, vg_to_oid_pred, ckn_oid_to_gt


def run_fast_contextual_openimages_inference(
    adapter: FastContextualAdapter,
    ckn_oid: Optional[FC_Net_OID],
    dataset: RelationDataset,
    vg_to_oid_obj: Dict[int, int],
    vg_to_oid_pred: Dict[int, int],
    ckn_oid_to_gt: Dict[int, int],
    variant_name: str,
    max_images: int = 200,
    top_k_rels: int = 100,
    cache_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached Fast Contextual SGG OpenImages predictions from {cache_path}...")
        cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return cached_data["predictions"][:max_images], cached_data["profile"]

    num_images = min(max_images, len(dataset))
    logger.info(f"Running Fast Contextual SGG ({variant_name}) inference across {num_images} OpenImages images...")

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

        if not os.path.exists(img_path):
            img_path = dataset.filenames[idx]

        pred_raw, timing = adapter.predict_image(img_path, top_k_rels=top_k_rels)

        boxes = pred_raw["boxes"]
        scores = pred_raw["pred_scores"]
        raw_labels = pred_raw["pred_labels"].numpy()
        raw_pairs = pred_raw["rel_pair_idxs"].numpy()
        raw_rel_scores = pred_raw["pred_rel_scores"].numpy()

        # Map object labels: VG150 ID -> OpenImages ID (0 if unmapped)
        mapped_labels = np.array([vg_to_oid_obj.get(int(l), 0) for l in raw_labels], dtype=np.int64)

        # Build 32-column relation score matrix (col 0: background, cols 1..31: OpenImages predicates)
        num_pairs = len(raw_pairs)
        mapped_rel_scores = np.zeros((num_pairs, 32), dtype=np.float32)

        if num_pairs > 0:
            # Method 1: Project VG CKN relation scores via ontology
            if len(raw_rel_scores) > 0:
                for v_pid, o_pid in vg_to_oid_pred.items():
                    if v_pid < raw_rel_scores.shape[1] and o_pid < 32:
                        mapped_rel_scores[:, o_pid] = np.maximum(
                            mapped_rel_scores[:, o_pid], raw_rel_scores[:, v_pid]
                        )

            # Method 2: If ckn_oid is provided, run OID CKN forward pass
            if ckn_oid is not None and len(boxes) >= 2:
                orig_h, orig_w = pred_raw["orig_size"]
                b_t = boxes
                norm_x1 = b_t[:, 0] / orig_w
                norm_y1 = b_t[:, 1] / orig_h
                norm_x2 = b_t[:, 2] / orig_w
                norm_y2 = b_t[:, 3] / orig_h
                cx = (norm_x2 - norm_x1) / 2.0
                cy = (norm_y2 - norm_y1) / 2.0
                loc_feat = torch.stack([cx, cy, norm_x1, norm_y1, norm_x2, norm_y2], dim=1).repeat(1, 5)

                raw_lbls_t = pred_raw["pred_labels"].long()
                word_feat = adapter.word_vec_list[raw_lbls_t]
                obj_f = torch.cat([word_feat, loc_feat], dim=1)

                p_t = pred_raw["rel_pair_idxs"]
                rel_f = torch.cat([
                    obj_f[p_t[:, 0]],
                    obj_f[p_t[:, 1]],
                    obj_f[p_t[:, 1]][:, 50:] - obj_f[p_t[:, 0]][:, 50:]
                ], dim=1)

                with torch.no_grad():
                    y_oid = ckn_oid(rel_f)
                    sig_oid = torch.sigmoid(y_oid)
                    conf_oid = sig_oid[:, 0:1]
                    sc_oid = (sig_oid[:, 1:] * conf_oid).cpu().numpy()

                for a_idx, o_pid in ckn_oid_to_gt.items():
                    if (a_idx - 1) < sc_oid.shape[1] and o_pid < 32:
                        mapped_rel_scores[:, o_pid] = np.maximum(
                            mapped_rel_scores[:, o_pid], sc_oid[:, a_idx - 1]
                        )

        pred_dict = {
            "image_id": img_id,
            "dataset_idx": idx,
            "orig_size": pred_raw["orig_size"],
            "boxes": boxes,
            "pred_scores": scores,
            "pred_labels": torch.from_numpy(mapped_labels),
            "raw_vg_labels": pred_raw["pred_labels"],
            "rel_pair_idxs": torch.from_numpy(raw_pairs),
            "pred_rel_scores": torch.from_numpy(mapped_rel_scores),
            "raw_vg_rel_scores": pred_raw["pred_rel_scores"],
        }

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
                f"[{variant_name}] Processed {idx + 1:3d}/{num_images:3d} images | "
                f"E2E: {avg_e2e:.1f} ms (Det: {avg_det:.1f} ms, Ctx: {avg_ctx:.1f} ms) | "
                f"RAM: {peak_ram:.1f} MB"
            )

    mean_det = float(np.mean(det_times))
    mean_ctx = float(np.mean(ctx_times))
    mean_e2e = float(np.mean(e2e_times))
    fps = 1000.0 / mean_e2e if mean_e2e > 0 else 0.0
    peak_mem = get_peak_memory_mb()

    profile = {
        "variant": variant_name,
        "mean_detector_latency_ms": mean_det,
        "mean_contextual_latency_ms": mean_ctx,
        "mean_e2e_latency_ms": mean_e2e,
        "fps": fps,
        "peak_rss_mb": peak_mem,
        "mean_candidate_pairs": float(np.mean(cand_pair_counts)),
    }

    if cache_path:
        torch.save({"predictions": predictions, "profile": profile}, cache_path)
        logger.info(f"Saved Fast Contextual SGG OpenImages predictions and profile to {cache_path}")

    return predictions, profile


def evaluate_fast_contextual_openimages_recalls(
    predictions: List[Dict[str, Any]],
    dataset: RelationDataset,
    k_list: List[int] = [50, 100]
) -> Dict[str, Any]:
    """
    Evaluates Scene Graph Detection (SGDet) Recall and Mean Recall on OpenImages V6 ground truth.
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


def evaluate_detection_mAP(predictions: List[Dict[str, Any]], dataset: RelationDataset) -> Tuple[float, float]:
    """
    Computes standard object detection mAP@50 and F1@50 on OpenImages V6 using COCOeval.
    """
    logger.info("Computing detection mAP@50 and F1@50 for Fast Contextual SGG on OpenImages V6...")
    images = []
    anns = []
    ann_id = 0

    for idx in range(len(predictions)):
        img_info = dataset.get_img_info(idx)
        images.append({
            "id": idx,
            "width": int(img_info.get("width", 1024)),
            "height": int(img_info.get("height", 1024)),
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
            # Only evaluate detected boxes whose mapped class is a valid positive OpenImages class
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
    parser = argparse.ArgumentParser(description="Fast Contextual SGG Benchmark on OpenImages V6 Dataset")
    parser.add_argument("--variant", type=str, default="C-SGG", choices=["C-SGG", "CV-SGG"], help="Model variant")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu or cuda)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"Starting Fast Contextual SGG ({args.variant}) Benchmark Pipeline on OpenImages V6 Dataset")
    logger.info(f"Max Images: {args.max_images} | Device: {args.device}")
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

    vg_to_oid_obj, vg_to_oid_pred, ckn_oid_to_gt = build_ontology_mappings()

    adapter = FastContextualAdapter(
        variant=args.variant,
        device=args.device
    )

    # Load OpenImages CKN model
    ckn_oid_weights = os.path.join(WORKSPACE_ROOT, "model_repos/fast-context-scene-graph-generation/OID_dataset/ckn/ckn_oid_PBS.pt")
    ckn_oid = None
    if os.path.exists(ckn_oid_weights):
        logger.info(f"Loading native OpenImages CKN weights from {ckn_oid_weights}...")
        ckn_oid = FC_Net_OID("OID")
        chk = torch.load(ckn_oid_weights, map_location=args.device, weights_only=False)
        m = chk["model"] if "model" in chk else chk
        ckn_oid.load_state_dict(m)
        ckn_oid.to(args.device).eval()

    mode_str = "smoke" if args.max_images <= 5 else "full"
    cache_file = os.path.join(CACHE_DIR, f"raw_preds_OpenImages_{args.variant.lower()}_{mode_str}.pt")

    predictions, profile = run_fast_contextual_openimages_inference(
        adapter=adapter,
        ckn_oid=ckn_oid,
        dataset=oid_dataset,
        vg_to_oid_obj=vg_to_oid_obj,
        vg_to_oid_pred=vg_to_oid_pred,
        ckn_oid_to_gt=ckn_oid_to_gt,
        variant_name=args.variant,
        max_images=args.max_images,
        cache_path=cache_file
    )

    # Detection metrics
    map_50, f1_50 = evaluate_detection_mAP(predictions, oid_dataset)
    logger.info(f"Fast Contextual SGG ({args.variant}) Object Detection on OpenImages V6: mAP@50 = {map_50:.4f} | F1@50 = {f1_50:.4f}")

    # SGG Recall metrics
    sgg_metrics = evaluate_fast_contextual_openimages_recalls(predictions, oid_dataset, k_list=[50, 100])

    r50 = sgg_metrics["recalls_eligible"]["R@50"]
    r100 = sgg_metrics["recalls_eligible"]["R@100"]
    mr50 = sgg_metrics["mean_recalls"]["mR@50"]
    mr100 = sgg_metrics["mean_recalls"]["mR@100"]
    latency_ms = profile["mean_e2e_latency_ms"]
    fps = profile["fps"]

    speed_str = f"{latency_ms:.1f} ms ({fps:.2f} FPS)"

    logger.info("=" * 80)
    logger.info(f"FINAL BENCHMARK SCORES FOR FAST CONTEXTUAL SGG ({args.variant}) ON OPENIMAGES V6:")
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
        "model": f"Fast Contextual SGG ({args.variant})",
        "variant": args.variant,
        "dataset": "OpenImages V6",
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

    metrics_json_path = os.path.join(LOG_DIR, "openimages_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    logger.info(f"Saved Fast Contextual SGG OpenImages metrics to {metrics_json_path}")

    metrics_csv_path = os.path.join(LOG_DIR, "openimages_metrics.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "R@50", "R@100", "mR@50", "mR@100", "F1@50", "Detection mAP@50", "Speed"])
        writer.writerow([
            f"Fast Contextual SGG ({args.variant})",
            "OpenImages V6",
            f"{r50*100:.2f}%",
            f"{r100*100:.2f}%",
            f"{mr50*100:.2f}%",
            f"{mr100*100:.2f}%",
            f"{f1_50*100:.2f}%",
            f"{map_50*100:.2f}%",
            speed_str
        ])
    logger.info(f"Saved Fast Contextual SGG OpenImages metrics CSV to {metrics_csv_path}")

    # Print requested table format
    print("\n" + "=" * 80)
    print("REQUESTED BENCHMARK OUTPUT TABLE:")
    print("R@50\tR@100\tmR@50\tmR@100\tF1@50\tDetection mAP@50\tSpeed")
    print(f"{r50*100:.2f}%\t{r100*100:.2f}%\t{mr50*100:.2f}%\t{mr100*100:.2f}%\t{f1_50*100:.2f}%\t{map_50*100:.2f}%\t{speed_str}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
