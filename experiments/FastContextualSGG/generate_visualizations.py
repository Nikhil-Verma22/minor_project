import os
import json
import cv2
import torch
import numpy as np

WORKSPACE_ROOT = "d:/Minor"
COCO_PATH = os.path.join(WORKSPACE_ROOT, "evaluation_sets/VG150_200_annotations.coco.json")
OUT_DIR = os.path.join(WORKSPACE_ROOT, "results/FastContextualSGG/visualizations")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_visualizations_for_variant(variant: str):
    pred_path = os.path.join(WORKSPACE_ROOT, f".cache/FastContextualSGG/raw_preds_VG150_{variant.lower()}_full.pt")
    if not os.path.exists(pred_path):
        print(f"Predictions file not found: {pred_path}")
        return

    with open(COCO_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    rel_names = {rel["id"]: rel["name"] for rel in coco["rel_categories"]}

    data = torch.load(pred_path, map_location="cpu")
    predictions = {p["image_id"]: p for p in data["predictions"]}

    target_image_ids = [2315564, 2315563]

    colors = [
        (255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0),
        (238, 130, 238), (0, 255, 255), (255, 140, 0), (147, 112, 219)
    ]

    for img_id in target_image_ids:
        if img_id not in predictions:
            continue

        pred = predictions[img_id]
        img_file = os.path.join(WORKSPACE_ROOT, f"evaluation_sets/images/VG150/{img_id}.jpg")
        if not os.path.exists(img_file):
            continue

        img = cv2.imread(img_file)
        vis = img.copy()

        boxes = pred["boxes"].numpy()
        scores = pred["pred_scores"].numpy()
        labels = pred["pred_labels"].numpy()
        rel_pairs = pred["rel_pair_idxs"].numpy()
        rel_labels = pred["pred_rel_labels"].numpy()
        rel_scores = pred["pred_rel_scores"].numpy()

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

            panel_y = 22
            cv2.putText(
                vis,
                f"{variant} Predicted Scene Graph (Top {len(top_rels)} Relations):",
                (10, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            panel_y += 24

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

                edge_color = colors[r_idx % len(colors)]
                cv2.arrowedLine(vis, s_center, o_center, edge_color, 2, tipLength=0.03)

        out_file = os.path.join(OUT_DIR, f"vg_{img_id}_{variant.lower()}_pred.jpg")
        cv2.imwrite(out_file, vis)
        print(f"Saved {variant} visualization: {out_file}")


def main():
    for variant in ["C-SGG", "CV-SGG"]:
        generate_visualizations_for_variant(variant)


if __name__ == "__main__":
    main()
