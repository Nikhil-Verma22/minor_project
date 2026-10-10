import os
import sys
import time
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Dict, Any, List, Tuple, Optional

WORKSPACE_ROOT = "d:/Minor"
VG_DIR = os.path.join(WORKSPACE_ROOT, "model_repos/fast-context-scene-graph-generation/VG_dataset")
if VG_DIR not in sys.path:
    sys.path.insert(0, VG_DIR)

from ckn.CKN import FC_Net
from vdn.VDN import visual_rel_model
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords


class FastContextualAdapter:
    """
    Standardized adapter for Fast Contextual Scene Graph Generation.
    Supports separate evaluation of:
      1. C-SGG: Contextual-only relationship prediction via Context Knowledge Network (CKN).
      2. CV-SGG: Contextual filtering followed by visual verification via Vision Differentiation Network (VDN).
    Preserves directed endpoints, official coordinate conventions, and outputs native VG150 distributions.
    """

    def __init__(
        self,
        variant: str = "C-SGG",
        detector_weights: str = "checkpoints/FastContextualSGG/yolol_object_test_28.pt",
        ckn_weights: str = "model_repos/fast-context-scene-graph-generation/VG_dataset/ckn/CKN+EPBS.pt",
        vdn_weights: str = "model_repos/fast-context-scene-graph-generation/VG_dataset/vdn/CKN+VDN+EPBS.pt",
        word_vec_path: str = ".cache/FastContextualSGG/vg_word_vec_list.pt",
        device: str = "cpu",
        conf_thres: float = 0.01,
        iou_thres: float = 0.45,
        obj_count_limit: int = 80,
        cv_filter_k: int = 150,
    ):
        self.variant = variant.upper()
        if self.variant not in ("C-SGG", "CV-SGG"):
            raise ValueError(f"Unknown variant '{variant}'. Supported: 'C-SGG', 'CV-SGG'")

        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.obj_count_limit = obj_count_limit
        self.cv_filter_k = cv_filter_k

        # 1. Load Object Detector (YOLOv5l)
        det_path = os.path.join(WORKSPACE_ROOT, detector_weights)
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Detector checkpoint not found at: {det_path}")
        det_chk = torch.load(det_path, map_location=self.device, weights_only=False)
        self.detector = det_chk["model"].float().to(self.device).eval()

        # 2. Load CKN Model
        ckn_path = os.path.join(WORKSPACE_ROOT, ckn_weights)
        if not os.path.exists(ckn_path):
            raise FileNotFoundError(f"CKN checkpoint not found at: {ckn_path}")
        self.ckn_model = FC_Net("VG")
        ckn_chk = torch.load(ckn_path, map_location=self.device)["model"]
        self.ckn_model.load_state_dict(ckn_chk)
        self.ckn_model.to(self.device).eval()

        # 3. Load VDN Model if CV-SGG
        self.vdn_model = None
        if self.variant == "CV-SGG":
            vdn_path = os.path.join(WORKSPACE_ROOT, vdn_weights)
            if not os.path.exists(vdn_path):
                raise FileNotFoundError(f"VDN checkpoint not found at: {vdn_path}")
            self.vdn_model = visual_rel_model("VG")
            vdn_chk = torch.load(vdn_path, map_location=self.device)["model"]
            self.vdn_model.load_state_dict(vdn_chk)
            self.vdn_model.to(self.device).eval()

        # 4. Load Word Vectors
        w_path = os.path.join(WORKSPACE_ROOT, word_vec_path)
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Word vectors not found at: {w_path}")
        self.word_vec_list = torch.load(w_path, map_location=self.device)

        # Image transforms for VDN crops
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def predict_image(
        self,
        img_path: str,
        top_k_rels: int = 100
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Runs full pipeline inference on a single image.
        Returns:
            pred_dict: {
                "orig_size": (H, W),
                "boxes": Tensor [N, 4] (x1, y1, x2, y2),
                "pred_scores": Tensor [N],
                "pred_labels": Tensor [N] (1-indexed 1..150),
                "rel_pair_idxs": Tensor [M, 2] (subj_idx, obj_idx),
                "pred_rel_scores": Tensor [M, 51] (0 is bg, 1..50 are VG predicates),
                "pred_rel_labels": Tensor [M] (1-indexed 1..50)
            }
            timing_dict: {
                "t_detector_ms": float,
                "t_contextual_ms": float,
                "t_visual_ms": float,
                "t_e2e_ms": float,
                "num_candidate_pairs": int,
                "num_verified_pairs": int
            }
        """
        t0_e2e = time.perf_counter()

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        orig_h, orig_w = bgr.shape[:2]

        # Stage 1: Object Detection
        t0_det = time.perf_counter()
        img_resized, ratio, (dw, dh) = letterbox(bgr, 640, stride=32, auto=True)
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            det_out = self.detector(img_tensor, augment=False)[0]

        det = non_max_suppression(
            det_out,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=None,
            agnostic=False
        )[0]
        t1_det = time.perf_counter()
        t_detector_ms = (t1_det - t0_det) * 1000.0

        if len(det) > 0:
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], (orig_h, orig_w)).round()
            det = det[det[:, 4].sort(descending=True)[1]]
            if len(det) > self.obj_count_limit:
                det = det[:self.obj_count_limit]

        # Check minimum objects required for relationships
        if len(det) < 2:
            t1_e2e = time.perf_counter()
            boxes = det[:, :4].cpu() if len(det) > 0 else torch.zeros((0, 4))
            scores = det[:, 4].cpu() if len(det) > 0 else torch.zeros(0)
            labels = det[:, 5].long().cpu() if len(det) > 0 else torch.zeros(0, dtype=torch.long)
            return {
                "orig_size": (orig_h, orig_w),
                "boxes": boxes,
                "pred_scores": scores,
                "pred_labels": labels,
                "rel_pair_idxs": torch.zeros((0, 2), dtype=torch.long),
                "pred_rel_scores": torch.zeros((0, 51), dtype=torch.float32),
                "pred_rel_labels": torch.zeros(0, dtype=torch.long),
            }, {
                "t_detector_ms": t_detector_ms,
                "t_contextual_ms": 0.0,
                "t_visual_ms": 0.0,
                "t_e2e_ms": (t1_e2e - t0_e2e) * 1000.0,
                "num_candidate_pairs": 0,
                "num_verified_pairs": 0,
            }

        boxes = det[:, :4]
        obj_scores = det[:, 4]
        obj_labels = det[:, 5].long()
        N = len(det)

        # Stage 2: Spatial & Semantic Context Construction
        norm_x1 = boxes[:, 0] / orig_w
        norm_y1 = boxes[:, 1] / orig_h
        norm_x2 = boxes[:, 2] / orig_w
        norm_y2 = boxes[:, 3] / orig_h
        cx = (norm_x2 - norm_x1) / 2.0
        cy = (norm_y2 - norm_y1) / 2.0
        loc_feat = torch.stack([cx, cy, norm_x1, norm_y1, norm_x2, norm_y2], dim=1)
        word_feat = self.word_vec_list[obj_labels]
        obj = torch.cat([word_feat, loc_feat.repeat(1, 5)], dim=1)

        obj_id = torch.arange(N, device=self.device)
        a_id = obj_id.repeat(N, 1).view(-1)
        b_id = torch.repeat_interleave(obj_id, N, dim=0)
        obj_pair = torch.cat([a_id.unsqueeze(1), b_id.unsqueeze(1)], dim=1)
        obj_pair = obj_pair[obj_pair[:, 0] != obj_pair[:, 1]]

        rel_feature = torch.cat([
            obj[obj_pair[:, 0]],
            obj[obj_pair[:, 1]],
            obj[obj_pair[:, 1]][:, 50:] - obj[obj_pair[:, 0]][:, 50:]
        ], dim=1)

        total_pairs = len(obj_pair)

        # Stage 3: Contextual Prediction via CKN
        t0_ctx = time.perf_counter()
        with torch.no_grad():
            y_ckn = self.ckn_model(rel_feature)
            pred_ckn = torch.sigmoid(y_ckn)
            ckn_conf = pred_ckn[:, 0:1]
            ckn_scores = pred_ckn[:, 1:] * ckn_conf
            pred_ckn_full = torch.cat([torch.zeros(total_pairs, 1, device=self.device), ckn_scores], dim=1)
            rel_max_ckn, rel_argmax_ckn = torch.max(pred_ckn_full, dim=1)
        t1_ctx = time.perf_counter()
        t_contextual_ms = (t1_ctx - t0_ctx) * 1000.0

        t_visual_ms = 0.0
        num_verified_pairs = 0

        if self.variant == "C-SGG":
            # C-SGG outputs purely contextual predictions
            pair_conf = obj_scores[obj_pair[:, 0]] * obj_scores[obj_pair[:, 1]] * rel_max_ckn
            sort_order = torch.argsort(pair_conf, descending=True)
            if top_k_rels is not None and len(sort_order) > top_k_rels:
                sort_order = sort_order[:top_k_rels]

            final_pairs = obj_pair[sort_order].cpu()
            final_scores = pred_ckn_full[sort_order].cpu()
            final_labels = rel_argmax_ckn[sort_order].cpu()

        else:
            # CV-SGG: Contextual candidate filtering followed by VDN visual verification
            K = min(self.cv_filter_k, total_pairs)
            filter_order = torch.argsort(rel_max_ckn, descending=True)[:K]
            cand_pairs = obj_pair[filter_order]
            cand_rel_feature = rel_feature[filter_order]
            num_verified_pairs = K

            t0_vis = time.perf_counter()
            visual_bbox_pair = torch.cat([boxes[cand_pairs[:, 0]], boxes[cand_pairs[:, 1]]], dim=1)
            cut_x1, _ = torch.min(visual_bbox_pair[:, [0, 4]], dim=1)
            cut_y1, _ = torch.min(visual_bbox_pair[:, [1, 5]], dim=1)
            cut_x2, _ = torch.max(visual_bbox_pair[:, [2, 6]], dim=1)
            cut_y2, _ = torch.max(visual_bbox_pair[:, [3, 7]], dim=1)

            batch_img = []
            for i in range(K):
                x1, y1 = max(0, int(cut_x1[i])), max(0, int(cut_y1[i]))
                x2, y2 = min(orig_w, int(cut_x2[i])), min(orig_h, int(cut_y2[i]))
                if x2 <= x1: x2 = x1 + 1
                if y2 <= y1: y2 = y1 + 1

                sx1, sy1 = max(0, int(visual_bbox_pair[i, 0])), max(0, int(visual_bbox_pair[i, 1]))
                sx2, sy2 = min(orig_w, int(visual_bbox_pair[i, 2])), min(orig_h, int(visual_bbox_pair[i, 3]))
                ox1, oy1 = max(0, int(visual_bbox_pair[i, 4])), max(0, int(visual_bbox_pair[i, 5]))
                ox2, oy2 = min(orig_w, int(visual_bbox_pair[i, 6])), min(orig_h, int(visual_bbox_pair[i, 7]))

                crop_rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)

                s_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
                s_mask[sy1:sy2, sx1:sx2] = 1.0
                s_mask_crop = cv2.resize(s_mask[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_AREA)

                o_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
                o_mask[oy1:oy2, ox1:ox2] = 1.0
                o_mask_crop = cv2.resize(o_mask[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_AREA)

                t_crop = transforms.ToTensor()(crop_rgb)
                t_crop = self.normalize(t_crop)
                s_t = torch.from_numpy(s_mask_crop).unsqueeze(0)
                o_t = torch.from_numpy(o_mask_crop).unsqueeze(0)

                patch = torch.cat([s_t, t_crop, o_t], dim=0)
                batch_img.append(patch)

            # Split into chunks of 50 for VDN forward pass
            batch_visual_pred = []
            batch_semantic_pred = []
            chunk_size = 50
            for start_idx in range(0, K, chunk_size):
                end_idx = min(start_idx + chunk_size, K)
                chunk_img = torch.stack(batch_img[start_idx:end_idx]).to(self.device)
                chunk_feat = cand_rel_feature[start_idx:end_idx]

                with torch.no_grad():
                    _, sem_pred, fusion_pred = self.vdn_model(chunk_img, chunk_feat)
                    batch_semantic_pred.append(sem_pred)
                    batch_visual_pred.append(fusion_pred)

            pred_semantic = torch.cat(batch_semantic_pred, dim=0)
            pred_fusion = torch.cat(batch_visual_pred, dim=0)

            # Apply official masking: retain top-10 semantic predicates, zero others
            sem_conf, sem_scores = pred_semantic[:, 0:1], pred_semantic[:, 1:]
            sem_full = torch.cat([torch.zeros(K, 1, device=self.device), sem_scores], dim=1)
            _, sem_indices = torch.sort(sem_full, 1, descending=True)
            indices_1d = torch.arange(K, device=self.device)
            neg_indices = torch.repeat_interleave(indices_1d, sem_indices[:, 10:].shape[1], dim=0)
            neg_pairs = torch.cat([neg_indices.unsqueeze(1), sem_indices[:, 10:].reshape(-1, 1)], dim=1)

            vis_scores = pred_fusion[:, 1:]
            vis_full = torch.cat([torch.zeros(K, 1, device=self.device), vis_scores], dim=1)
            vis_full[neg_pairs[:, 0], neg_pairs[:, 1]] = 0.0
            vis_full = vis_full * sem_conf

            rel_max_cv, rel_argmax_cv = torch.max(vis_full, dim=1)
            t1_vis = time.perf_counter()
            t_visual_ms = (t1_vis - t0_vis) * 1000.0

            cv_pair_conf = obj_scores[cand_pairs[:, 0]] * obj_scores[cand_pairs[:, 1]] * rel_max_cv
            sort_order = torch.argsort(cv_pair_conf, descending=True)
            if top_k_rels is not None and len(sort_order) > top_k_rels:
                sort_order = sort_order[:top_k_rels]

            final_pairs = cand_pairs[sort_order].cpu()
            final_scores = vis_full[sort_order].cpu()
            final_labels = rel_argmax_cv[sort_order].cpu()

        t1_e2e = time.perf_counter()
        t_e2e_ms = (t1_e2e - t0_e2e) * 1000.0

        pred_dict = {
            "orig_size": (orig_h, orig_w),
            "boxes": boxes.cpu(),
            "pred_scores": obj_scores.cpu(),
            "pred_labels": obj_labels.cpu(),
            "rel_pair_idxs": final_pairs,
            "pred_rel_scores": final_scores,
            "pred_rel_labels": final_labels,
        }

        timing_dict = {
            "t_detector_ms": t_detector_ms,
            "t_contextual_ms": t_contextual_ms,
            "t_visual_ms": t_visual_ms,
            "t_e2e_ms": t_e2e_ms,
            "num_candidate_pairs": total_pairs,
            "num_verified_pairs": num_verified_pairs,
        }

        return pred_dict, timing_dict
