#!/usr/bin/env python3
"""Convert HaMeR hand metadata (*.data.pyd) to COCO wholebody-hand style JSON.

Parallel over files with --workers N. Output is deterministic for a given
seed regardless of worker count: files are shuffled once with the seed, and
image/annotation IDs are assigned sequentially in that order after conversion.
"""
import argparse
import multiprocessing as mp
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
from PIL import Image
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
    image_h=None,
) -> List[float]:
    w = scale[0] * 200 * pad_scale
    h = scale[1] * 200 * pad_scale
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
    metadata_path: Path,
    input_dir: Path,
    pad_scale: float,
    reorder: List[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert one metadata file. IDs are placeholders (0); assigned later."""
    with open(metadata_path, "rb") as f:
        people = pickle.load(f)

    img_path = find_matching_image(metadata_path)
    with Image.open(img_path) as im:
        image_w, image_h = im.size

    image = {
        "id": 0,
        "file_name": str(img_path.relative_to(input_dir)),
        "width": image_w,
        "height": image_h,
    }

    anns: List[Dict[str, Any]] = []

    for person in people:
        if "keypoints_2d" not in person:
            raise ValueError(f"{metadata_path}: missing 'keypoints_2d'")

        points_2d = person["keypoints_2d"]
        points_2d = maybe_reorder_points(points_2d, reorder=reorder)

        if len(points_2d) != 21:
            raise ValueError(f"{metadata_path}: expected 21 landmarks, got {len(points_2d)}")

        hand_kpts = flatten_keypoints_xy_to_coco(points_2d)
        hand_box = compute_bbox_from_center_and_scale(
            person["center"], person["scale"],
            pad_scale=pad_scale, image_w=image_w, image_h=image_h,
        )
        left_valid, left_box, left_kpts = build_empty_hand()
        right_valid, right_box, right_kpts = build_empty_hand()

        hand_side = person["right"]
        if hand_side < 0.5:
            left_valid, left_box, left_kpts = 1, hand_box, hand_kpts
        elif hand_side >= 0.5:
            right_valid, right_box, right_kpts = 1, hand_box, hand_kpts
        else:
            raise ValueError("hand_side must be '0' or '1'")

        ann = {
            "id": 0,
            "image_id": 0,
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

    return image, anns


# ---- worker pool plumbing ------------------------------------------------

_WORKER_CTX: Dict[str, Any] = {}


def _init_worker(input_dir: Path, pad_scale: float, reorder: Optional[List[int]]) -> None:
    _WORKER_CTX["input_dir"] = input_dir
    _WORKER_CTX["pad_scale"] = pad_scale
    _WORKER_CTX["reorder"] = reorder


def _convert_safe(metadata_path: Path):
    """Returns (image, anns, None) on success, (None, None, error_str) on failure."""
    try:
        image, anns = convert_one_file(
            metadata_path=metadata_path,
            input_dir=_WORKER_CTX["input_dir"],
            pad_scale=_WORKER_CTX["pad_scale"],
            reorder=_WORKER_CTX["reorder"],
        )
        return image, anns, None
    except Exception as e:  # noqa: BLE001 - skip-and-warn, matches prior behavior
        return None, None, f"{type(e).__name__}: {e}"


def iter_converted(files, input_dir, pad_scale, reorder, workers):
    """Yield (path, image, anns, err) in the order of `files`."""
    if workers <= 1:
        _init_worker(input_dir, pad_scale, reorder)
        for p in files:
            yield (p, *_convert_safe(p))
        return

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(input_dir, pad_scale, reorder),
    ) as pool:
        for p, result in zip(files, pool.imap(_convert_safe, files, chunksize=64)):
            yield (p, *result)


# ---------------------------------------------------------------------------


def load_reorder_mapping(reorder_str: str) -> Optional[List[int]]:
    """Parse comma-separated mapping like "0,13,14,..." """
    if not reorder_str:
        return None
    vals = [int(x.strip()) for x in reorder_str.split(",")]
    if len(vals) != 21:
        raise ValueError("Reorder mapping must contain exactly 21 integers")
    return vals


def main():
    parser = argparse.ArgumentParser(description="Convert HaMeR hand metadata to COCO-style JSON")
    parser.add_argument("--input-dir", required=True, help="Directory containing *.data.pyd files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for the COCO-style annotation JSON (default: <input-dir>/annotations)")
    parser.add_argument("--description", required=True, help="Description for the COCO-style annotation JSON")
    parser.add_argument("--pad-scale", type=float, default=1, help="Padding scale for bbox generation")
    parser.add_argument("--reorder", type=str, default="",
                        help="Optional 21-index reorder mapping, comma-separated")
    parser.add_argument("--seed", type=int, default=709, help="Seed used to shuffle the files")
    parser.add_argument("--size", type=int, default=-1,
                        help="Max number of annotations to convert (-1 for all)")
    parser.add_argument("--glob", default="*.data.pyd", help="Glob pattern for metadata files")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_train_json = out_dir / "coco_annotations.json"

    reorder = load_reorder_mapping(args.reorder)

    metadata_files = sorted(input_dir.rglob(args.glob))
    if not metadata_files:
        raise FileNotFoundError(f"No files matched {args.glob} in {input_dir}")
    random.Random(args.seed).shuffle(metadata_files)

    n = len(metadata_files) if args.size < 0 else min(len(metadata_files), args.size)
    metadata_files = metadata_files[:n]

    print(f"Found {len(metadata_files)} metadata files, workers={args.workers}")

    coco = {
        "info": {
            "description": f"{args.description}",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [build_category()],
    }

    image_id = 1
    ann_id = 1

    results = iter_converted(metadata_files, input_dir, args.pad_scale, reorder, args.workers)
    for meta_path, image, anns, err in tqdm(results, total=len(metadata_files), desc="convert"):
        if err is not None:
            print(f"[WARN] Skipping {meta_path}: {err}")
            continue
        if args.size > 0 and len(coco["annotations"]) + len(anns) > args.size:
            continue

        image["id"] = image_id
        for i, ann in enumerate(anns):
            ann["id"] = ann_id + i
            ann["image_id"] = image_id

        coco["images"].append(image)
        coco["annotations"].extend(anns)
        image_id += 1
        ann_id += len(anns)

        if args.size > 0 and len(coco["annotations"]) >= args.size:
            break

    with output_train_json.open("wb") as f:
        f.write(orjson.dumps(coco))

    print(f"Wrote {output_train_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")


if __name__ == "__main__":
    main()
