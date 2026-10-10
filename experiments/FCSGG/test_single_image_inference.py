import os
import sys
import time
import cv2
import json
import torch
import types
import numpy as np
import PIL.Image

# Compatibility shims
PIL.Image.LINEAR = PIL.Image.BILINEAR
if "detectron2._C" not in sys.modules:
    m = types.ModuleType("detectron2._C")
    sys.modules["detectron2._C"] = m

WORKSPACE_ROOT = "d:/Minor"
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fcsgg/detectron2"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "model_repos/fcsgg"))

from detectron2.config import get_cfg
from fcsgg.config import add_fcsgg_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

def test_inference():
    cfg_file = os.path.join(WORKSPACE_ROOT, "model_repos/fcsgg/configs/FCSGG_HRNet_W32_2xDownRAF_512x512_MS.yaml")
    ckpt_file = os.path.join(WORKSPACE_ROOT, "checkpoints/FCSGG/HRNetW32-1S.pth")
    img_file = os.path.join(WORKSPACE_ROOT, "evaluation_sets/images/VG150/2315564.jpg")
    
    print(f"Loading config from {cfg_file}...")
    cfg = get_cfg()
    add_fcsgg_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.HRNET.WEIGHTS = ""
    cfg.MODEL.WEIGHTS = ckpt_file
    
    print("Building model...")
    model = build_model(cfg)
    model.eval()
    
    print(f"Loading checkpoint from {ckpt_file}...")
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(ckpt_file)
    
    print(f"Reading image {img_file}...")
    bgr = cv2.imread(img_file)
    orig_h, orig_w = bgr.shape[:2]
    print(f"Original image shape: {bgr.shape}")
    
    aug = T.ResizeShortestEdge(short_edge_length=(512, 512), max_size=512)
    transform = aug.get_transform(bgr)
    resized_bgr = transform.apply_image(bgr)
    print(f"Resized image shape: {resized_bgr.shape}")
    
    tensor_img = torch.as_tensor(resized_bgr.transpose(2, 0, 1).astype("float32"))
    batched_inputs = [{
        "image": tensor_img,
        "height": orig_h,
        "width": orig_w
    }]
    
    print("Running forward pass...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(batched_inputs)
    t1 = time.perf_counter()
    
    print(f"Inference complete in {(t1 - t0)*1000:.2f} ms")
    print(f"Outputs type: {type(outputs)}, len: {len(outputs)}")
    out0 = outputs[0]
    print(f"Keys in output[0]: {list(out0.keys())}")
    
    instances = out0.get("instances", None)
    scene_graph = out0.get("scene_graph", None)
    
    if instances is not None:
        print(f"Instances fields: {instances.get_fields().keys()}")
        pred_boxes = instances.pred_boxes.tensor
        scores = instances.scores
        pred_classes = instances.pred_classes
        print(f"Detected {len(pred_boxes)} boxes.")
        if len(pred_boxes) > 0:
            print(f"Sample box 0: {pred_boxes[0].tolist()}, score: {scores[0].item():.4f}, class: {pred_classes[0].item()}")
            
    if scene_graph is not None:
        print(f"SceneGraph fields: {dir(scene_graph)}")
        pred_graph = scene_graph.pred_graph if hasattr(scene_graph, "pred_graph") else None
        rel_scores = scene_graph.rel_scores if hasattr(scene_graph, "rel_scores") else None
        rel_inds = scene_graph.rel_inds if hasattr(scene_graph, "rel_inds") else None
        print(f"pred_graph shape: {pred_graph.shape if pred_graph is not None else None}")
        print(f"rel_scores shape: {rel_scores.shape if rel_scores is not None else None}")
        print(f"rel_inds shape: {rel_inds.shape if rel_inds is not None else None}")
        if pred_graph is not None and len(pred_graph) > 0:
            print(f"Sample triplet 0: subj_cls={pred_graph[0, 0].item()}, obj_cls={pred_graph[0, 1].item()}, pred={pred_graph[0, 2].item()}, score={pred_graph[0, 5].item():.4f}")

if __name__ == "__main__":
    test_inference()
