#!/usr/bin/env python3
"""Split the HaMeR mpiinzsl-train COCO annotation file into real/in-the-wild
(hand_labels) and synthetic (hand_labels_synth) subsets, classified by the
subfolder in each image's file_name.

Usage:
    python split_mpiinzsl_annotations.py data/hamer/mpiinzsl-train/annotations/coco_annotations.json

Writes coco_annotations_wild.json and coco_annotations_synth.json next to the
input (or to --out-dir). Exits nonzero if any image matches neither prefix.
"""
import argparse
import json
from pathlib import Path


def classify(file_name: str) -> str:
    p = file_name.replace("\\", "/")
    if "hand_labels_synth" in p:
        return "synth"
    if "hand_labels" in p:
        return "wild"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ann_file", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    coco = json.loads(args.ann_file.read_text())
    out_dir = args.out_dir or args.ann_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img_bucket = {}  # image_id -> bucket
    images = {"wild": [], "synth": []}
    unknown = []
    for img in coco["images"]:
        b = classify(img["file_name"])
        if b == "unknown":
            unknown.append(img["file_name"])
            continue
        img_bucket[img["id"]] = b
        images[b].append(img)

    anns = {"wild": [], "synth": []}
    orphans = 0
    for ann in coco["annotations"]:
        b = img_bucket.get(ann["image_id"])
        if b is None:
            orphans += 1
            continue
        anns[b].append(ann)

    for bucket in ("wild", "synth"):
        out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
        out["images"] = images[bucket]
        out["annotations"] = anns[bucket]
        path = out_dir / f"coco_annotations_{bucket}.json"
        path.write_text(json.dumps(out, separators=(",", ":")))
        print(f"{path}: {len(images[bucket]):,} images, {len(anns[bucket]):,} annotations")

    print(f"input : {len(coco['images']):,} images, {len(coco['annotations']):,} annotations")
    print(f"split : {len(images['wild']) + len(images['synth']):,} images, "
          f"{len(anns['wild']) + len(anns['synth']):,} annotations")

    ok = True
    if unknown:
        ok = False
        print(f"ERROR: {len(unknown)} images matched neither prefix, e.g.:")
        for f in unknown[:10]:
            print("   ", f)
    if orphans:
        ok = False
        print(f"ERROR: {orphans} annotations reference unclassified images")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()