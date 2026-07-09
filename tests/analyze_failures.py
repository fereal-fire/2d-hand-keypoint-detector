import os
import json
import pickle
import argparse
from collections import defaultdict

import cv2
import numpy as np


HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def normalize_path(path):
    return path.replace("\\", "/")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def xywh_to_xyxy(box):
    x, y, w, h = np.asarray(box, dtype=np.float32).flatten()[:4]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def cs_to_xyxy(box):
    """
    Convert MMPose center/scale box to xyxy.

    MMPose top-down boxes are commonly:
    [center_x, center_y, scale_x, scale_y, area, score]

    scale is normalized by 200.
    """
    cx, cy, sx, sy = np.asarray(box, dtype=np.float32).flatten()[:4]
    w = sx * 200.0
    h = sy * 200.0

    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=np.float32,
    )


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box[:4]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_size(box):
    x1, y1, x2, y2 = box[:4]
    return max(1.0, max(x2 - x1, y2 - y1))


def normalized_center_distance(pred_box, gt_box):
    pc = box_center(pred_box)
    gc = box_center(gt_box)
    size = box_size(gt_box)
    return float(np.linalg.norm(pc - gc) / size)


def bbox_from_keypoints(kpts, pad=10):
    kpts = np.asarray(kpts)
    xs = kpts[:, 0]
    ys = kpts[:, 1]

    if kpts.shape[1] >= 3:
        valid = (kpts[:, 2] > 0) & (xs > 0) & (ys > 0)
    else:
        valid = (xs > 0) & (ys > 0)

    if valid.sum() == 0:
        return None

    x1 = float(xs[valid].min()) - pad
    y1 = float(ys[valid].min()) - pad
    x2 = float(xs[valid].max()) + pad
    y2 = float(ys[valid].max()) + pad

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def draw_gt_hand(
    img,
    kpts,
    visible_color=(0, 255, 0),      # green
    occluded_color=(0, 255, 255),   # yellow
    radius=3,
    thickness=2,
    draw_occluded=True,
):
    """
    COCO visibility:
      v = 0: not labeled
      v = 1: labeled but occluded
      v = 2: visible

    Draws:
      visible GT = green
      occluded/labeled GT = yellow
    """
    pts = np.asarray(kpts)

    def joint_drawable(i):
        x, y = pts[i, 0], pts[i, 1]

        if x <= 0 or y <= 0:
            return False

        if pts.shape[1] < 3:
            return True

        v = pts[i, 2]

        if v == 2:
            return True

        if v == 1:
            return draw_occluded

        return False

    def joint_color(i):
        if pts.shape[1] >= 3 and pts[i, 2] == 1:
            return occluded_color
        return visible_color

    for a, b in HAND_SKELETON:
        if not joint_drawable(a) or not joint_drawable(b):
            continue

        xa, ya = pts[a, 0], pts[a, 1]
        xb, yb = pts[b, 0], pts[b, 1]

        color = visible_color
        if pts.shape[1] >= 3:
            if pts[a, 2] == 1 or pts[b, 2] == 1:
                color = occluded_color

        cv2.line(
            img,
            (int(xa), int(ya)),
            (int(xb), int(yb)),
            color,
            thickness,
        )

    for i in range(min(21, len(pts))):
        if not joint_drawable(i):
            continue

        x, y = pts[i, 0], pts[i, 1]
        cv2.circle(
            img,
            (int(x), int(y)),
            radius,
            joint_color(i),
            -1,
        )


def draw_pred_hand(
    img,
    kpts,
    color=(0, 0, 255),  # red
    radius=3,
    thickness=2,
    min_score=0.0,
):
    """
    Prediction third column is confidence score, not COCO visibility.
    """
    pts = np.asarray(kpts)

    def pred_drawable(i):
        x, y = pts[i, 0], pts[i, 1]

        if x <= 0 or y <= 0:
            return False

        if pts.shape[1] >= 3:
            return pts[i, 2] > min_score

        return True

    for a, b in HAND_SKELETON:
        if pred_drawable(a) and pred_drawable(b):
            xa, ya = pts[a, 0], pts[a, 1]
            xb, yb = pts[b, 0], pts[b, 1]

            cv2.line(
                img,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                color,
                thickness,
            )

    for i in range(min(21, len(pts))):
        if pred_drawable(i):
            x, y = pts[i, 0], pts[i, 1]
            cv2.circle(img, (int(x), int(y)), radius, color, -1)


def draw_bbox(img, box, color, thickness=2):
    if box is None:
        return

    x1, y1, x2, y2 = box[:4]

    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness,
    )


def crop_union(img, box_a, box_b, pad=60):
    h, w = img.shape[:2]

    x1 = int(max(0, min(box_a[0], box_b[0]) - pad))
    y1 = int(max(0, min(box_a[1], box_b[1]) - pad))
    x2 = int(min(w, max(box_a[2], box_b[2]) + pad))
    y2 = int(min(h, max(box_a[3], box_b[3]) + pad))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def build_gt_hand_instances(coco):
    """
    Explodes COCO-WholeBody person annotations into separate hand instances.

    Each output item:
      file_name
      image_id
      ann_id
      hand_type: left/right/generic
      gt_kpts
      gt_bbox
    """
    images_by_id = {img["id"]: img for img in coco["images"]}
    hand_instances_by_file = defaultdict(list)

    for ann in coco["annotations"]:
        img = images_by_id[ann["image_id"]]
        file_name = normalize_path(img["file_name"])

        # COCO-WholeBody left hand
        if "lefthand_kpts" in ann and len(ann["lefthand_kpts"]) == 63:
            kpts = np.asarray(ann["lefthand_kpts"], dtype=np.float32).reshape(21, 3)

            if np.any(kpts[:, 2] > 0):
                if "lefthand_box" in ann and ann["lefthand_box"] is not None:
                    box = xywh_to_xyxy(ann["lefthand_box"])
                else:
                    box = bbox_from_keypoints(kpts)
                # box = bbox_from_keypoints(kpts)

                if box is not None:
                    hand_instances_by_file[file_name].append({
                        "file_name": file_name,
                        "image_id": ann["image_id"],
                        "ann_id": ann.get("id"),
                        "hand_type": "left",
                        "gt_kpts": kpts,
                        "gt_bbox": box,
                    })

        # COCO-WholeBody right hand
        if "righthand_kpts" in ann and len(ann["righthand_kpts"]) == 63:
            kpts = np.asarray(ann["righthand_kpts"], dtype=np.float32).reshape(21, 3)

            if np.any(kpts[:, 2] > 0):
                if "righthand_box" in ann and ann["righthand_box"] is not None:
                    box = xywh_to_xyxy(ann["righthand_box"])
                else:
                    box = bbox_from_keypoints(kpts)
                # box = bbox_from_keypoints(kpts)

                if box is not None:
                    hand_instances_by_file[file_name].append({
                        "file_name": file_name,
                        "image_id": ann["image_id"],
                        "ann_id": ann.get("id"),
                        "hand_type": "right",
                        "gt_kpts": kpts,
                        "gt_bbox": box,
                    })

        # Generic 21-keypoint hand annotation format
        if "keypoints" in ann and len(ann["keypoints"]) == 63:
            kpts = np.asarray(ann["keypoints"], dtype=np.float32).reshape(21, 3)

            if np.any(kpts[:, 2] > 0):
                if "bbox" in ann and ann["bbox"] is not None:
                    box = xywh_to_xyxy(ann["bbox"])
                else:
                    box = bbox_from_keypoints(kpts)

                if box is not None:
                    hand_instances_by_file[file_name].append({
                        "file_name": file_name,
                        "image_id": ann["image_id"],
                        "ann_id": ann.get("id"),
                        "hand_type": "generic",
                        "gt_kpts": kpts,
                        "gt_bbox": box,
                    })

    return hand_instances_by_file


def extract_pred_records(preds_raw):
    records = []

    if not isinstance(preds_raw, list):
        raise ValueError(f"Unsupported prediction pickle type: {type(preds_raw)}")

    for item in preds_raw:
        if not isinstance(item, dict):
            continue

        if "preds" not in item:
            continue

        preds = np.asarray(item["preds"])
        boxes = np.asarray(item["boxes"]) if item.get("boxes") is not None else None
        image_paths = item.get("image_paths", [])
        bbox_ids = item.get("bbox_ids", [])

        for i in range(len(preds)):
            records.append({
                "pred": np.asarray(preds[i]),
                "box_raw": np.asarray(boxes[i]) if boxes is not None else None,
                "image_path": image_paths[i] if i < len(image_paths) else None,
                "bbox_id": bbox_ids[i] if i < len(bbox_ids) else None,
            })

    return records


def find_file_match(image_path, gt_by_file):
    if image_path is None:
        return None

    image_path = normalize_path(image_path)
    base = os.path.basename(image_path)

    # GT file_name may be just "000000252219.jpg"
    if base in gt_by_file:
        return base

    # Or it may include a prefix like "val2017/000000252219.jpg"
    matches = [
        fn for fn in gt_by_file.keys()
        if image_path.endswith(fn) or fn.endswith(base)
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        matches.sort(key=len, reverse=True)
        return matches[0]

    return None


def get_pred_bbox_xyxy(rec):
    box = rec.get("box_raw")

    if box is not None and len(box) >= 6:
        # Confirmed from your debug:
        # [center_x, center_y, scale_x, scale_y, area, score]
        return cs_to_xyxy(box)

    if box is not None and len(box) >= 4:
        x1, y1, x2, y2 = box[:4]

        if x2 > x1 and y2 > y1:
            return np.array([x1, y1, x2, y2], dtype=np.float32)

    return bbox_from_keypoints(rec["pred"], pad=10)


def safe_float_for_csv(x):
    if x is None or np.isnan(x):
        return "nan"
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--min-iou", type=float, default=0.1)
    parser.add_argument("--max-center-dist", type=float, default=None)
    parser.add_argument("--sort-by", default="visible", choices=["visible", "labeled", "occluded"])
    parser.add_argument("--pred-min-score", type=float, default=0.0)
    parser.add_argument("--debug-first", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    crop_dir = os.path.join(args.out_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)

    with open(args.pred, "rb") as f:
        preds_raw = pickle.load(f)

    if args.debug_first:
        print("Prediction pickle type:", type(preds_raw))
        print("Prediction pickle length:", len(preds_raw) if hasattr(preds_raw, "__len__") else "NA")

        first = preds_raw[0]
        print("First item type:", type(first))

        if isinstance(first, dict):
            print("First item keys:", first.keys())

            for k, v in first.items():
                print("\nKEY:", k)
                print("TYPE:", type(v))
                try:
                    print("LEN:", len(v))
                except Exception:
                    pass
                print("SAMPLE:", str(v)[:1000])

        return

    coco = load_json(args.ann)
    gt_by_file = build_gt_hand_instances(coco)
    pred_records = extract_pred_records(preds_raw)

    scored = []
    no_file_match = 0
    no_gt_candidates = 0
    low_iou = 0
    high_center_dist = 0

    for rec in pred_records:
        pred = np.asarray(rec["pred"])

        if pred.ndim != 2 or pred.shape[0] != 21:
            continue

        matched_file = find_file_match(rec["image_path"], gt_by_file)

        if matched_file is None:
            no_file_match += 1
            continue

        candidates = gt_by_file.get(matched_file, [])

        if not candidates:
            no_gt_candidates += 1
            continue

        pred_box = get_pred_bbox_xyxy(rec)

        if pred_box is None:
            continue

        best = None
        best_iou = -1.0

        for gt in candidates:
            cur_iou = iou_xyxy(pred_box, gt["gt_bbox"])

            if cur_iou > best_iou:
                best_iou = cur_iou
                best = gt

        if best is None:
            continue

        if best_iou < args.min_iou:
            low_iou += 1
            continue

        center_dist = normalized_center_distance(pred_box, best["gt_bbox"])

        if args.max_center_dist is not None and center_dist > args.max_center_dist:
            high_center_dist += 1
            continue

        gt_kpts = best["gt_kpts"]

        labeled_mask = gt_kpts[:, 2] > 0
        visible_mask = gt_kpts[:, 2] == 2
        occluded_mask = gt_kpts[:, 2] == 1

        if labeled_mask.sum() == 0:
            continue

        diff_labeled = pred[labeled_mask, :2] - gt_kpts[labeled_mask, :2]
        epe_labeled = np.linalg.norm(diff_labeled, axis=1)

        mean_epe_labeled = float(epe_labeled.mean())
        max_epe_labeled = float(epe_labeled.max())

        if visible_mask.sum() > 0:
            diff_visible = pred[visible_mask, :2] - gt_kpts[visible_mask, :2]
            epe_visible = np.linalg.norm(diff_visible, axis=1)
            mean_epe_visible = float(epe_visible.mean())
            max_epe_visible = float(epe_visible.max())
        else:
            mean_epe_visible = float("nan")
            max_epe_visible = float("nan")

        if occluded_mask.sum() > 0:
            diff_occluded = pred[occluded_mask, :2] - gt_kpts[occluded_mask, :2]
            epe_occluded = np.linalg.norm(diff_occluded, axis=1)
            mean_epe_occluded = float(epe_occluded.mean())
            max_epe_occluded = float(epe_occluded.max())
        else:
            mean_epe_occluded = float("nan")
            max_epe_occluded = float("nan")

        scored.append({
            "mean_epe_labeled": mean_epe_labeled,
            "max_epe_labeled": max_epe_labeled,
            "mean_epe_visible": mean_epe_visible,
            "max_epe_visible": max_epe_visible,
            "mean_epe_occluded": mean_epe_occluded,
            "max_epe_occluded": max_epe_occluded,
            "num_labeled": int(labeled_mask.sum()),
            "num_visible": int(visible_mask.sum()),
            "num_occluded": int(occluded_mask.sum()),
            "file_name": matched_file,
            "ann_id": best["ann_id"],
            "hand_type": best["hand_type"],
            "match_iou": float(best_iou),
            "center_dist": float(center_dist),
            "pred": pred,
            "gt": gt_kpts,
            "pred_bbox": pred_box,
            "gt_bbox": best["gt_bbox"],
            "image_path": rec["image_path"],
            "bbox_id": rec["bbox_id"],
        })

    def sort_value(row):
        if args.sort_by == "visible":
            val = row["mean_epe_visible"]
        elif args.sort_by == "occluded":
            val = row["mean_epe_occluded"]
        else:
            val = row["mean_epe_labeled"]

        if np.isnan(val):
            return -1.0

        return val

    scored.sort(key=sort_value, reverse=True)

    summary_path = os.path.join(args.out_dir, "worst_samples.csv")

    with open(summary_path, "w") as f:
        f.write(
            "rank,"
            "mean_epe_labeled,max_epe_labeled,"
            "mean_epe_visible,max_epe_visible,"
            "mean_epe_occluded,max_epe_occluded,"
            "num_labeled,num_visible,num_occluded,"
            "match_iou,center_dist,"
            "ann_id,hand_type,bbox_id,file_name,"
            "gt_bbox_xyxy,pred_bbox_xyxy\n"
        )

        for rank, row in enumerate(scored[:args.top_k], start=1):
            f.write(
                f"{rank},"
                f"{row['mean_epe_labeled']:.3f},"
                f"{row['max_epe_labeled']:.3f},"
                f"{safe_float_for_csv(row['mean_epe_visible'])},"
                f"{safe_float_for_csv(row['max_epe_visible'])},"
                f"{safe_float_for_csv(row['mean_epe_occluded'])},"
                f"{safe_float_for_csv(row['max_epe_occluded'])},"
                f"{row['num_labeled']},"
                f"{row['num_visible']},"
                f"{row['num_occluded']},"
                f"{row['match_iou']:.4f},"
                f"{row['center_dist']:.4f},"
                f"{row['ann_id']},"
                f"{row['hand_type']},"
                f"{row['bbox_id']},"
                f"{row['file_name']},"
                f"\"{row['gt_bbox'].tolist()}\","
                f"\"{row['pred_bbox'].tolist()}\"\n"
            )

            img_path = row["image_path"]

            if img_path is None or not os.path.exists(img_path):
                img_path = os.path.join(args.img_root, os.path.basename(row["file_name"]))

            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            vis = img.copy()

            # GT visible = green
            # GT occluded/labeled = yellow
            # Prediction = red
            draw_gt_hand(
                vis,
                row["gt"],
                visible_color=(0, 255, 0),
                occluded_color=(0, 255, 255),
                radius=3,
                thickness=2,
                draw_occluded=True,
            )

            draw_pred_hand(
                vis,
                row["pred"],
                color=(0, 0, 255),
                radius=3,
                thickness=2,
                min_score=args.pred_min_score,
            )

            # GT bbox = green rectangle
            # Pred/input bbox = red rectangle
            draw_bbox(vis, row["gt_bbox"], color=(0, 255, 0), thickness=2)
            draw_bbox(vis, row["pred_bbox"], color=(0, 0, 255), thickness=2)

            text = (
                f"rank={rank} "
                f"EPE_lab={row['mean_epe_labeled']:.1f} "
                f"EPE_vis={safe_float_for_csv(row['mean_epe_visible'])} "
                f"iou={row['match_iou']:.2f} "
                f"cd={row['center_dist']:.2f} "
                f"vis={row['num_visible']} "
                f"occ={row['num_occluded']} "
                f"{row['hand_type']}"
            )

            cv2.putText(
                vis,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                vis,
                text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            safe_name = os.path.basename(row["file_name"])

            out_name = (
                f"{rank:04d}_"
                f"epevis_{safe_float_for_csv(row['mean_epe_visible'])}_"
                f"epelab_{row['mean_epe_labeled']:.1f}_"
                f"iou_{row['match_iou']:.2f}_"
                f"cd_{row['center_dist']:.2f}_"
                f"{row['hand_type']}_{safe_name}"
            )

            out_path = os.path.join(args.out_dir, out_name)
            cv2.imwrite(out_path, vis)

            crop = crop_union(vis, row["gt_bbox"], row["pred_bbox"], pad=60)

            if crop is not None:
                crop_out_path = os.path.join(crop_dir, out_name)
                cv2.imwrite(crop_out_path, crop)

    print("")
    print("Done.")
    print(f"GT files with hand instances: {len(gt_by_file)}")
    print(f"Prediction records extracted: {len(pred_records)}")
    print(f"Scored predictions: {len(scored)}")
    print(f"No file match: {no_file_match}")
    print(f"No GT candidates: {no_gt_candidates}")
    print(f"Skipped for low IoU < {args.min_iou}: {low_iou}")

    if args.max_center_dist is not None:
        print(f"Skipped for center_dist > {args.max_center_dist}: {high_center_dist}")

    print(f"Sort by: {args.sort_by}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved visualizations to: {args.out_dir}")
    print(f"Saved crops to: {crop_dir}")
    print("")
    print("Color legend:")
    print("  Green  = GT visible joints, v=2")
    print("  Yellow = GT occluded/labeled joints, v=1")
    print("  Red    = model prediction")
    print("  Green box = GT hand bbox")
    print("  Red box   = prediction/input bbox")


if __name__ == "__main__":
    main()