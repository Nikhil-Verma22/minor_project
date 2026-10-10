import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, "d:/Minor/model_repos/SGG-Benchmark")
from sgg_benchmark.config import get_cfg
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.utils.checkpoint import DetectronCheckpointer
from sgg_benchmark.structures.image_list import to_image_list

def letterbox(bgr: np.ndarray, size: int = 640):
    h, w = bgr.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    pad_w = (size - nw) / 2
    pad_h = (size - nh) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, gain, pad_w, pad_h

def main():
    cfg_raw = OmegaConf.load("model_repos/SGG-Benchmark/configs/hydra/PSG/REACT.yaml")
    with open_dict(cfg_raw):
        cfg_raw.model.device = "cpu"
        cfg_raw.output_dir = "checkpoints/REACT/react_PSG/react_PSG"
        cfg_raw.glove_dir = "d:/Minor/.cache/REACT/glove"
        cfg_raw.model.roi_box_head.num_classes = 134
        cfg_raw.model.roi_box_head.mlp_head_dim = 512
        cfg_raw.model.roi_box_head.pooler_scales = [0.0625]
        cfg_raw.model.roi_relation_head.mlp_head_dim = 512
        cfg_raw.model.roi_relation_head.num_classes = 57
    cfg = get_cfg(cfg_raw)
    
    print("Building model...")
    model = build_detection_model(cfg)
    model.eval()
    
    ckpt_path = "checkpoints/REACT/react_PSG/react_PSG/best_model_epoch_11.pth"
    print(f"Loading checkpoint {ckpt_path}...")
    checkpointer = DetectronCheckpointer(cfg, model)
    checkpointer.load(ckpt_path)
    
    img_path = "evaluation_sets/images/PSG/107907.jpg"
    print(f"Reading image {img_path}...")
    bgr = cv2.imread(img_path)
    orig_h, orig_w = bgr.shape[:2]
    padded, gain, pad_w, pad_h = letterbox(bgr, 640)
    
    # BGR to RGB, normalize
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    # GeneralizedYOLO expects pixel mean/std if specified, but YOLO backbone takes [0, 1] tensor
    # Let's check GeneralizedYOLO forward
    image_list = to_image_list([tensor])
    
    print("Running forward pass...")
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(image_list)
    t1 = time.perf_counter()
    
    print(f"Inference complete in {(t1 - t0)*1000:.2f} ms")
    print(f"Output type: {type(output)}, length: {len(output)}")
    res0 = output[0]
    print(f"Keys in output[0]: {list(res0.keys())}")
    
    boxes = res0.get("boxes", None)
    scores = res0.get("pred_scores", None)
    labels = res0.get("pred_labels", None)
    rel_pair_idxs = res0.get("rel_pair_idxs", None)
    pred_rel_scores = res0.get("pred_rel_scores", None)
    
    print(f"Boxes detected: {len(boxes) if boxes is not None else 0}")
    if boxes is not None and len(boxes) > 0:
        print(f"Sample boxes (first 3): {boxes[:3]}")
        print(f"Sample scores: {scores[:3]}")
        print(f"Sample labels: {labels[:3]}")
    
    print(f"Candidate pairs: {len(rel_pair_idxs) if rel_pair_idxs is not None else 0}")
    if rel_pair_idxs is not None and len(rel_pair_idxs) > 0:
        print(f"Sample pair indices (first 3): {rel_pair_idxs[:3]}")
        print(f"Pred rel scores shape: {pred_rel_scores.shape}")
        
    print("Single image inference test PASSED!")

if __name__ == "__main__":
    main()
