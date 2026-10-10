import os
import sys
import time
import types
import cv2
import torch
import numpy as np
import PIL.Image
from typing import Dict, Any, List, Tuple, Optional

# Ensure Pillow 10+ and Detectron2 compatibility
PIL.Image.LINEAR = PIL.Image.BILINEAR
if "detectron2._C" not in sys.modules:
    m = types.ModuleType("detectron2._C")
    sys.modules["detectron2._C"] = m

WORKSPACE_ROOT = "d:/Minor"
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fcsgg/detectron2"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fcsgg"))

from detectron2.config import get_cfg
from fcsgg.config import add_fcsgg_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

class FCSGGAdapter:
    """
    Standardized adapter for the pretrained FCSGG (HRNetW32-1S) model.
    Preserves original CenterNet box coordinates, RAF relationship field scores,
    subject/object directions, and outputs native VG150 predictions.
    """

    def __init__(
        self,
        config_path: str = "model_repos/fcsgg/configs/FCSGG_HRNet_W32_2xDownRAF_512x512_MS.yaml",
        checkpoint_path: str = "checkpoints/FCSGG/HRNetW32-1S.pth",
        device: str = "cpu"
    ):
        self.device = device
        self.config_path = os.path.join(WORKSPACE_ROOT, config_path)
        self.checkpoint_path = os.path.join(WORKSPACE_ROOT, checkpoint_path)
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"FCSGG checkpoint not found: {self.checkpoint_path}")
            
        cfg = get_cfg()
        add_fcsgg_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.HRNET.WEIGHTS = ""
        cfg.MODEL.WEIGHTS = self.checkpoint_path
        self.cfg = cfg

        # Build model and load checkpoint
        self.model = build_model(cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.checkpoint_path)

        # Preprocessing transform: resize shortest edge to 512, max 512
        self.aug = T.ResizeShortestEdge(short_edge_length=(512, 512), max_size=512)

    def predict_image(self, img_path: str, top_k_rels: int = 100) -> Tuple[Dict[str, Any], float, float]:
        """
        Runs inference on a single image and converts output to standardized format.
        Returns:
            pred_dict: {
                "boxes": Tensor [N, 4] (x1, y1, x2, y2 in orig coordinates),
                "pred_scores": Tensor [N],
                "pred_labels": Tensor [N] (1-indexed VG150 classes 1..150),
                "rel_pair_idxs": Tensor [M, 2] (subj_idx, obj_idx),
                "pred_rel_scores": Tensor [M, 51] (0 is bg, 1..50 are native VG predicates),
                "pred_rel_labels": Tensor [M] (1-indexed VG150 predicates 1..50)
            }
            fwd_latency_ms: float
            e2e_latency_ms: float
        """
        t0_e2e = time.perf_counter()
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        orig_h, orig_w = bgr.shape[:2]

        transform = self.aug.get_transform(bgr)
        resized_bgr = transform.apply_image(bgr)

        tensor_img = torch.as_tensor(resized_bgr.transpose(2, 0, 1).astype("float32")).to(self.device)
        batched_inputs = [{
            "image": tensor_img,
            "height": orig_h,
            "width": orig_w
        }]

        t0_fwd = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(batched_inputs)
        t1_fwd = time.perf_counter()

        out = outputs[0]
        instances = out["instances"]
        scene_graph = out.get("scene_graph", None)

        # 1. Boxes and Object detections
        boxes_raw = instances.pred_boxes.tensor.detach().cpu()
        scores_raw = instances.scores.detach().cpu()
        classes_raw = instances.pred_classes.detach().cpu()

        # In CenterNet, classes are 0-indexed (0..149). Visual Genome canonical classes are 1..150.
        classes_vg = classes_raw + 1

        # 2. Relations and RAF scores
        if scene_graph is not None and hasattr(scene_graph, "rel_scores") and hasattr(scene_graph, "rel_inds"):
            rel_scores_raw = scene_graph.rel_scores.detach().cpu()  # [M, 50]
            rel_inds_raw = scene_graph.rel_inds.detach().cpu()      # [M, 2]

            # Top-k filter if needed
            if len(rel_scores_raw) > top_k_rels:
                max_sc = rel_scores_raw.max(dim=1).values
                top_order = torch.argsort(max_sc, descending=True)[:top_k_rels]
                rel_scores_raw = rel_scores_raw[top_order]
                rel_inds_raw = rel_inds_raw[top_order]

            M = len(rel_scores_raw)
            # Create standard 51-dim relation score matrix (col 0: background, cols 1..50: native predicates)
            full_rel_scores = torch.zeros((M, 51), dtype=torch.float32)
            if M > 0:
                full_rel_scores[:, 1:] = rel_scores_raw
                pred_rel_labels = torch.argmax(rel_scores_raw, dim=1) + 1
            else:
                pred_rel_labels = torch.zeros(0, dtype=torch.long)
        else:
            rel_inds_raw = torch.zeros((0, 2), dtype=torch.long)
            full_rel_scores = torch.zeros((0, 51), dtype=torch.float32)
            pred_rel_labels = torch.zeros(0, dtype=torch.long)

        pred_dict = {
            "orig_size": (orig_h, orig_w),
            "boxes": boxes_raw,
            "pred_scores": scores_raw,
            "pred_labels": classes_vg,
            "rel_pair_idxs": rel_inds_raw,
            "pred_rel_scores": full_rel_scores,
            "pred_rel_labels": pred_rel_labels,
        }

        t1_e2e = time.perf_counter()
        fwd_latency_ms = (t1_fwd - t0_fwd) * 1000.0
        e2e_latency_ms = (t1_e2e - t0_e2e) * 1000.0

        return pred_dict, fwd_latency_ms, e2e_latency_ms
