#!/usr/bin/env python3
import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image
import pickle
import numpy as np
import random
import orjson
from tqdm import tqdm


def find_matching_image(pyd_path: Path) -> Path:
    stem = pyd_path.name.replace(".data.pyd", "")
    for ext in (".jpg", ".jpeg", ".png"):
        p = pyd_path.with_name(stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for {pyd_path}")


def flatten_keypoints_xy_to_coco(points_2d: List[List[float]]) -> List[float]:
    if len(points_2d) != 21:
        raise ValueError(f"Expected 21 keypoints, got {len(points_2d)}")

    out = []
    for pt in points_2d:
        if len(pt) != 3:
            raise ValueError(f"Each 2D point must have length 3, got {pt}")
        x, y, v = pt
        if v == 1:
            v = 2
        out.extend([float(x), float(y), int(v)])
    return out

def compute_bbox_from_center_and_scale(
    center: List[float],
    scale: List[float],
    pad_scale: float = 1.0,
    image_w=None,
    image_h=None
) -> List[float]:
    w = scale[0] * pad_scale
    h = scale[1] * pad_scale
    x = center[0] - w / 2
    y = center[1] - h / 2
    if image_w is not None:
        if x < 0:
            w += x
            x = 0.0
        w = min(w, image_w - x)
    if image_h is not None:
        if y < 0:
            h += y
            y = 0.0
        h = min(h, image_h - y)

    return [float(x), float(y), float(max(w, 1.0)), float(max(h, 1.0))]

def build_category() -> Dict[str, Any]:
    return {
        "id": 1,
        "name": "person",
    }


def build_empty_hand() -> Tuple[int, List[float], List[float]]:
    return 0, [0.0, 0.0, 0.0, 0.0], [0.0] * 63


def maybe_reorder_points(points_2d: List[List[float]], reorder: List[int] = None) -> List[List[float]]:
    if reorder is None:
        return points_2d

    if len(reorder) != 21:
        raise ValueError("reorder must have length 21")

    return [points_2d[i] for i in reorder]


def convert_one_file(
    metadata: Tuple[str, List[Dict[str, Any]]],   # CHANGED: (imgname, people) from npz, not a .pyd path
    image_id: int,
    ann_id: int,
    input_dir: Path,
    pad_scale: float,
    reorder: List[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    imgname, people = metadata                     # CHANGED: unpack npz-derived record

    img_path = input_dir / imgname                 # CHANGED: image path comes from npz imgname
    with Image.open(img_path) as im:
        image_w, image_h = im.size

    image = {
            "id": image_id,
            "file_name": str(imgname),             # CHANGED: imgname is already relative to input_dir
            "width": image_w,
            "height": image_h,
        }


    anns: List[Dict[str, Any]] = []

    for i, person in enumerate(people):
        if "keypoints_2d" not in person:
            raise ValueError(f"{imgname}: missing 'keypoints_2d'")

        points_2d = person["keypoints_2d"]
        points_2d = maybe_reorder_points(points_2d, reorder=reorder)

        if len(points_2d) != 21:
            raise ValueError(f"{imgname}: expected 21 landmarks, got {len(points_2d)}")

        hand_kpts = flatten_keypoints_xy_to_coco(points_2d)
        hand_box = compute_bbox_from_center_and_scale(person["center"], person["scale"], pad_scale=pad_scale, image_w=image_w, image_h=image_h,)
        left_valid, left_box, left_kpts = build_empty_hand()
        right_valid, right_box, right_kpts = build_empty_hand()

        hand_side = person["right"]
        if hand_side < .5:
            left_valid, left_box, left_kpts = 1, hand_box, hand_kpts
        elif hand_side >= .5:
            right_valid, right_box, right_kpts = 1, hand_box, hand_kpts
        else:
            raise ValueError("hand_side must be '0' or '1'")

        ann = {
            "id": ann_id + i,
            "image_id": image_id,
            "category_id": 1,
            "iscrowd": 0,
            "lefthand_box": left_box,
            "righthand_box": right_box,
            "lefthand_kpts": left_kpts,
            "righthand_kpts": right_kpts,
            "lefthand_valid": left_valid,
            "righthand_valid": right_valid,
        }
        anns.append(ann)

    return [image], anns


def load_npz_records(npz_path: Path) -> List[Tuple[str, List[Dict[str, Any]]]]:
    # CHANGED: replaces the *.data.pyd glob. Reads the HInt eval npz and groups
    # its rows by image so each record looks like one .pyd file (imgname, people).
    d = np.load(npz_path, allow_pickle=True)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in d["imgname"]]
    centers, scales, rights = d["center"], d["scale"], d["right"]
    kps = d["hand_keypoints_2d"]
    print(f"  sample scale[:3] = {scales[:3].tolist()}  (~100-300 => pixels; ~1-3 => /200)")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for i, name in enumerate(names):
        person = {
            "keypoints_2d": kps[i].tolist(),
            "center": centers[i].tolist(),
            "scale": scales[i].tolist(),
            "right": float(rights[i]),
        }
        groups.setdefault(name, []).append(person)
    return list(groups.items())


def load_reorder_mapping(reorder_str: str) -> List[int]:
    """
    Parse comma-separated mapping like:
      "0,13,14,..."
    """
    if not reorder_str:
        return None
    vals = [int(x.strip()) for x in reorder_str.split(",")]
    if len(vals) != 21:
        raise ValueError("Reorder mapping must contain exactly 21 integers")
    return vals


def main():
    parser = argparse.ArgumentParser(description="Convert HaMeR hand metadata to COCO-style JSON")
    parser.add_argument("--npz", required=True, help="HInt eval npz, e.g. .../TEST_ego4d_img_all.npz") 
    parser.add_argument("--input-dir", required=True, help="IMG_DIR; images are relative to this") 
    parser.add_argument("--output-dir", required=False, help="Output COCO-style annotation JSON")
    parser.add_argument(
        "--pad-scale",
        type=float,
        default=1,
        help="Padding scale for bbox generation",
    )
    parser.add_argument(
        "--reorder",
        type=str,
        default="",
        help="Optional 21-index reorder mapping, comma-separated",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.npz).parent / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    reorder = load_reorder_mapping(args.reorder)

    metadata_files = load_npz_records(Path(args.npz)) 
    if not metadata_files:
        raise FileNotFoundError(f"No records found in {args.npz}")

    coco = {
        "info": {
            "description": "",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [build_category()],
    }

    image_id = 1
    ann_id = 1

    print("Converting training")
    for meta in tqdm(metadata_files, desc="train"):
        try:
            images, anns  = convert_one_file(
                metadata=meta,
                image_id=image_id,
                input_dir=input_dir,
                ann_id=ann_id,
                pad_scale=args.pad_scale,
                reorder=reorder,
            )

            coco["images"].extend(images)
            coco["annotations"].extend(anns)
            image_id += len(images)
            ann_id += len(anns)
        except Exception as e:
            print(f"[WARN] Skipping {meta[0]}: {e}")
    
    npz_stem = Path(args.npz).stem            # e.g. TEST_epick_img_occ
    output_train_json = out_dir / f"hint_{npz_stem}.json"
    output_train_json.parent.mkdir(parents=True, exist_ok=True)
    with output_train_json.open("w") as f:
        json.dump(coco, f)

    print(f"Wrote {output_train_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")

if __name__ == "__main__":
    main()
