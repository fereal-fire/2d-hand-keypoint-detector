#!/usr/bin/env python3
import os
import json
import math
import random
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Any

from tqdm import tqdm


def flatten_keypoints_xy_to_coco(points_2d: List[List[float]], visibility: int = 2) -> List[float]:
    """
    Convert [[x, y], ...] -> [x1, y1, v1, x2, y2, v2, ...]
    """
    if len(points_2d) != 21:
        raise ValueError(f"Expected 21 keypoints, got {len(points_2d)}")

    out = []
    for pt in points_2d:
        if len(pt) != 2:
            raise ValueError(f"Each 2D point must have length 2, got {pt}")
        x, y = pt
        out.extend([float(x), float(y), int(visibility)])
    return out


def compute_bbox_from_points(
    points_2d: List[List[float]],
    image_w: int,
    image_h: int,
    pad_scale: float = 1.25,
    min_size: float = 2.0,
) -> List[float]:
    """
    Compute [x, y, w, h] from keypoints, with padding and clipping to image bounds.
    """
    xs = [float(p[0]) for p in points_2d]
    ys = [float(p[1]) for p in points_2d]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    w = max(x_max - x_min, min_size)
    h = max(y_max - y_min, min_size)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    w *= pad_scale
    h *= pad_scale

    x = cx - w / 2.0
    y = cy - h / 2.0

    # clip to image bounds
    x = max(0.0, x)
    y = max(0.0, y)
    w = min(w, image_w - x)
    h = min(h, image_h - y)

    return [float(x), float(y), float(w), float(h)]


def metadata_to_image_filename(metadata_path: Path, image_ext: str) -> str:
    stem = metadata_path.stem
    if not stem.startswith("metadata_"):
        raise ValueError(f"Unexpected metadata filename: {metadata_path.name}")
    image_stem = stem.replace("metadata_", "img_", 1)
    return image_stem + image_ext


def build_category() -> Dict[str, Any]:
    return {
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": [
            "wrist",
            "forefinger1",
            "forefinger2",
            "forefinger3",
            "middle_finger1",
            "middle_finger2",
            "middle_finger3",
            "pinky_finger1",
            "pinky_finger2",
            "pinky_finger3",
            "ring_finger1",
            "ring_finger2",
            "ring_finger3",
            "thumb1",
            "thumb2",
            "thumb3",
            "forefinger4",
            "middle_finger4",
            "pinky_finger4",
            "ring_finger4",
            "thumb4",
        ],
        "skeleton": []
    }


def build_empty_hand() -> Tuple[int, List[float], List[float]]:
    return 0, [0.0, 0.0, 0.0, 0.0], [0.0] * 63


def maybe_reorder_points(points_2d: List[List[float]], reorder: List[int] = None) -> List[List[float]]:
    """
    If SynthMoCap joint order differs from your dataset_info order,
    pass a 21-length index mapping here.
    """
    if reorder is None:
        return points_2d

    if len(reorder) != 21:
        raise ValueError("reorder must have length 21")

    return [points_2d[i] for i in reorder]


def convert_one_file(
    metadata_path: Path,
    hand_side: str,
    image_ext: str,
    pad_scale: float,
    reorder: List[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert one metadata file. image/ann IDs are placeholders (0); assigned later."""
    with metadata_path.open("r") as f:
        meta = json.load(f)

    if "landmarks" not in meta or "2D" not in meta["landmarks"]:
        raise ValueError(f"{metadata_path}: missing landmarks['2D']")

    points_2d = meta["landmarks"]["2D"]
    points_2d = maybe_reorder_points(points_2d, reorder=reorder)

    if len(points_2d) != 21:
        raise ValueError(f"{metadata_path}: expected 21 landmarks, got {len(points_2d)}")

    resolution = meta.get("camera", {}).get("resolution", [512, 512])
    if len(resolution) != 2:
        raise ValueError(f"{metadata_path}: bad camera.resolution = {resolution}")

    image_w = int(resolution[0])
    image_h = int(resolution[1])

    hand_kpts = flatten_keypoints_xy_to_coco(points_2d, visibility=2)
    hand_box = compute_bbox_from_points(points_2d, image_w, image_h, pad_scale=pad_scale)

    image = {
        "id": 0,
        "file_name": metadata_to_image_filename(metadata_path, image_ext=image_ext),
        "width": image_w,
        "height": image_h,
        "license": 0,
        "coco_url": "",
        "date_captured": "",
        "flickr_url": "",
    }

    left_valid, left_box, left_kpts = build_empty_hand()
    right_valid, right_box, right_kpts = build_empty_hand()

    hand_side = hand_side.lower()
    if hand_side == "left":
        left_valid, left_box, left_kpts = 1, hand_box, hand_kpts
    elif hand_side == "right":
        right_valid, right_box, right_kpts = 1, hand_box, hand_kpts
    else:
        raise ValueError("hand_side must be 'left' or 'right'")

    ann = {
        "id": 0,
        "image_id": 0,
        "category_id": 1,
        "iscrowd": 0,
        "segmentation": [],
        "num_keypoints": 0,
        "area": float(hand_box[2] * hand_box[3]),
        "bbox": hand_box,
        "keypoints": [0.0] * 51,
        "face_box": [0.0, 0.0, 0.0, 0.0],
        "lefthand_box": left_box,
        "righthand_box": right_box,
        "lefthand_kpts": left_kpts,
        "righthand_kpts": right_kpts,
        "face_kpts": [],
        "face_valid": 0,
        "lefthand_valid": left_valid,
        "righthand_valid": right_valid,
        "foot_valid": 0,
        "foot_kpts": [],
    }

    return image, ann


# ---- worker pool plumbing --------------------------------------------------

_WORKER_CTX: Dict[str, Any] = {}


def _init_worker(hand_side: str, image_ext: str, pad_scale: float, reorder: List[int]) -> None:
    _WORKER_CTX.update(hand_side=hand_side, image_ext=image_ext, pad_scale=pad_scale, reorder=reorder)


def _convert_safe(metadata_path: Path):
    """Returns (image, ann, None) on success, (None, None, error_str) on failure."""
    try:
        image, ann = convert_one_file(
            metadata_path=metadata_path,
            hand_side=_WORKER_CTX["hand_side"],
            image_ext=_WORKER_CTX["image_ext"],
            pad_scale=_WORKER_CTX["pad_scale"],
            reorder=_WORKER_CTX["reorder"],
        )
        return image, ann, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def iter_converted(files, hand_side, image_ext, pad_scale, reorder, workers):
    """Yield (path, image, ann, err) in the order of `files`."""
    if workers <= 1:
        _init_worker(hand_side, image_ext, pad_scale, reorder)
        for p in files:
            yield (p, *_convert_safe(p))
        return

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(hand_side, image_ext, pad_scale, reorder),
    ) as pool:
        for p, result in zip(files, pool.imap(_convert_safe, files, chunksize=64)):
            yield (p, *result)


# -----------------------------------------------------------------------------


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
    parser = argparse.ArgumentParser(description="Convert SynthMoCap hand metadata to COCO-style JSON")
    parser.add_argument("--input-dir", required=True, help="Directory containing metadata_*.json files")
    parser.add_argument("--output-dir", required=False, help="Output COCO-style annotation JSON")
    parser.add_argument(
        "--hand-side",
        default="left",
        choices=["left", "right"],
        help="Which side to store the hand under in the annotation",
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        help="Image extension used to derive file_name from metadata filename",
    )
    parser.add_argument(
        "--pad-scale",
        type=float,
        default=1.25,
        help="Padding scale for bbox generation",
    )
    parser.add_argument(
        "--reorder",
        type=str,
        default="",
        help="Optional 21-index reorder mapping, comma-separated",
    )
    parser.add_argument(
        "--glob",
        default="metadata_*.json",
        help="Glob pattern for metadata files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=709,
        help="Seed used to shuffle the files before applying --size (default: 709, matches convert_hamer.py)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=-1,
        help="Max number of samples to convert (-1 for all). One annotation per image here, "
             "so this caps both images and annotations.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "annotations"
    reorder = load_reorder_mapping(args.reorder)

    metadata_files = sorted(input_dir.glob(args.glob))
    if not metadata_files:
        raise FileNotFoundError(f"No files matched {args.glob} in {input_dir}")

    random.Random(args.seed).shuffle(metadata_files)
    n = len(metadata_files) if args.size < 0 else min(len(metadata_files), args.size)
    metadata_files = metadata_files[:n]

    coco = {
        "info": {
            "description": "SynthMoCap hand converted to COCO-style format",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [build_category()],
    }

    image_id = 1
    ann_id = 1

    print(f"Found {len(metadata_files)} metadata files, workers={args.workers}")

    results = iter_converted(
        metadata_files, args.hand_side, args.image_ext, args.pad_scale, reorder, args.workers
    )
    for meta_path, image, ann, err in tqdm(results, total=len(metadata_files), desc="convert"):
        if err is not None:
            print(f"[WARN] Skipping {meta_path}: {err}")
            continue

        image["id"] = image_id
        ann["id"] = ann_id
        ann["image_id"] = image_id

        coco["images"].append(image)
        coco["annotations"].append(ann)
        image_id += 1
        ann_id += 1

    output_train_json = out_dir / "coco_annotations.json"
    output_train_json.parent.mkdir(parents=True, exist_ok=True)
    with output_train_json.open("w") as f:
        json.dump(coco, f)

    print(f"Wrote {output_train_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")


if __name__ == "__main__":
    main()
