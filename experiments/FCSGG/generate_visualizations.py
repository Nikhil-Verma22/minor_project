import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path

WORKSPACE_ROOT = "d:/Minor"
PRED_PATH = os.path.join(WORKSPACE_ROOT, ".cache/FCSGG/raw_preds_VG150_fcsgg_full.pt")
COCO_PATH = os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")
OUT_DIR = os.path.join(WORKSPACE_ROOT, "results/FCSGG/visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_visualizations():
    with open(COCO_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    rel_names = {rel["id"]: rel["name"] for rel in coco["rel_categories"]}

    data = torch.load(PRED_PATH, map_location="cpu")
    predictions = {p["image_id"]: p for p in data["predictions"]}

    target_image_ids = [2315564, 2315563]

    for img_id in target_image_ids:
        if img_id not in predictions:
            print(f"Image {img_id} not found in predictions!")
            continue

        pred = predictions[img_id]
        img_file = os.path.join(WORKSPACE_ROOT, f"evaluation_sets/images/VG150/{img_id}.jpg")
        if not os.path.exists(img_file):
            print(f"Image file {img_file} not found!")
            continue

        img = cv2.imread(img_file)
        vis = img.copy()

        boxes = pred["boxes"].numpy()
        scores = pred["pred_scores"].numpy()
        labels = pred["pred_labels"].numpy()
        rel_pairs = pred["rel_pair_idxs"].numpy()
        rel_labels = pred["pred_rel_labels"].numpy()
        rel_scores = pred["pred_rel_scores"].numpy()

        colors = [
            (255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0),
            (238, 130, 238), (0, 255, 255), (255, 140, 0), (147, 112, 219)
        ]

        # Filter boxes score > 0.15
        valid_box_indices = [i for i, s in enumerate(scores) if s >= 0.15]

        for i in valid_box_indices:
            box = boxes[i].astype(int)
            x1, y1, x2, y2 = box
            lbl_id = labels[i]
            lbl_name = cat_names.get(lbl_id, f"cls_{lbl_id}")
            score = scores[i]
            color = colors[i % len(colors)]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            caption = f"#{i} {lbl_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(vis, caption, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw top relationship triplets
        if len(rel_pairs) > 0:
            top_rel_indices = []
            for r_idx in range(len(rel_pairs)):
                s_i, o_i = rel_pairs[r_idx]
                r_lbl = rel_labels[r_idx]
                if s_i in valid_box_indices and o_i in valid_box_indices and r_lbl > 0:
                    r_sc = rel_scores[r_idx, r_lbl]
                    top_rel_indices.append((r_idx, s_i, o_i, r_lbl, r_sc))

            top_rel_indices.sort(key=lambda x: x[4], reverse=True)
            top_rels = top_rel_indices[:6]

            panel_y = 20
            cv2.putText(vis, f"FCSGG Predicted Scene Graph (Top {len(top_rels)} Relations):", 
                        (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            panel_y += 22

            for rank, (r_idx, s_i, o_i, r_lbl, r_sc) in enumerate(top_rels):
                s_name = cat_names.get(labels[s_i], f"#{s_i}")
                o_name = cat_names.get(labels[o_i], f"#{o_i}")
                p_name = rel_names.get(r_lbl, f"rel_{r_lbl}")

                rel_str = f"[{rank+1}] #{s_i} {s_name} --({p_name}: {r_sc:.2f})--> #{o_i} {o_name}"
                cv2.putText(vis, rel_str, (15, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                panel_y += 20

                s_box = boxes[s_i]
                o_box = boxes[o_i]
                s_center = (int((s_box[0] + s_box[2]) / 2), int((s_box[1] + s_box[3]) / 2))
                o_center = (int((o_box[0] + o_box[2]) / 2), int((o_box[1] + o_box[3]) / 2))

                cv2.arrowedLine(vis, s_center, o_center, (0, 255, 255), 2, tipLength=0.04)
                mid_point = (int((s_center[0] + o_center[0]) / 2), int((s_center[1] + o_center[1]) / 2))
                cv2.putText(vis, p_name, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        out_file = os.path.join(OUT_DIR, f"VG150_{img_id}_pred.jpg")
        cv2.imwrite(out_file, vis)
        print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    generate_visualizations()
