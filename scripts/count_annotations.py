#!/usr/bin/env python3
"""
Report image/annotation counts for every subset produced by
convert_all_annotations.sh / convert_all_datasets.sh, without regenerating
anything. Run this on the VM (repo root or anywhere, via --data-root).

Layout it expects (matches scripts/convert_all_datasets.sh):
  <DATA_ROOT>/hamer/<dataset>/annotations/coco_annotations.json
  <DATA_ROOT>/synthmocap/synth_hand/annotations/coco_synthmocap_annotation.json
  <DATA_ROOT>/annotations/hint_{newdays,epick,ego4d}_{all,vis,occ}.json

Usage:
  python scripts/count_annotations.py
  python scripts/count_annotations.py --data-root /path/to/data
  python scripts/count_annotations.py --format csv > subset_counts.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import orjson

    def _load(path: Path) -> Dict[str, Any]:
        return orjson.loads(path.read_bytes())
except ImportError:  # orjson not required just to count
    def _load(path: Path) -> Dict[str, Any]:
        with path.open("rb") as f:
            return json.load(f)


def count_coco_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return {n_images, n_annotations} for a COCO-style json, or None on error."""
    try:
        data = _load(path)
    except Exception as e:
        return {"error": str(e)}
    return {
        "n_images": len(data.get("images", [])),
        "n_annotations": len(data.get("annotations", [])),
    }


def find_hamer_subsets(data_root: Path) -> List[Dict[str, Any]]:
    hamer_root = data_root / "hamer"
    rows = []
    if not hamer_root.is_dir():
        return rows
    for dataset_dir in sorted(hamer_root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in {
            "raw_archives", "hamer_training_data", "annotations",
        }:
            continue
        json_path = dataset_dir / "annotations" / "coco_annotations.json"
        row = {"group": "hamer", "subset": dataset_dir.name, "path": json_path}
        if json_path.exists():
            row.update(count_coco_json(json_path))
        else:
            row["missing"] = True
        rows.append(row)
    return rows


def find_synthmocap_subset(data_root: Path) -> List[Dict[str, Any]]:
    json_path = (
        data_root / "synthmocap" / "synth_hand" / "annotations"
        / "coco_synthmocap_annotation.json"
    )
    row = {"group": "synthmocap", "subset": "synth_hand", "path": json_path}
    if json_path.exists():
        row.update(count_coco_json(json_path))
    else:
        row["missing"] = True
    return [row]


def find_hint_subsets(data_root: Path) -> List[Dict[str, Any]]:
    ann_dir = data_root / "annotations"
    rows = []
    for name in ("newdays", "epick", "ego4d"):
        for split in ("all", "vis", "occ"):
            json_path = ann_dir / f"hint_{name}_{split}.json"
            row = {"group": "hint", "subset": f"{name}_{split}", "path": json_path}
            if json_path.exists():
                row.update(count_coco_json(json_path))
            else:
                row["missing"] = True
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data dir (default: ./data, matches DATA_ROOT in convert_all_datasets.sh)",
    )
    parser.add_argument(
        "--format", choices=["table", "csv"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="Don't print rows for subsets that haven't been converted yet",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    rows = (
        find_hamer_subsets(data_root)
        # + find_synthmocap_subset(data_root)
        # + find_hint_subsets(data_root)
    )

    if args.skip_missing:
        rows = [r for r in rows if not r.get("missing")]

    if args.format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["group", "subset", "n_images", "n_annotations", "status", "path"])
        for r in rows:
            status = "missing" if r.get("missing") else r.get("error", "ok")
            writer.writerow([
                r["group"], r["subset"],
                r.get("n_images", ""), r.get("n_annotations", ""),
                status, r["path"],
            ])
        return 0

    print(f"Data root: {data_root}\n")
    header = f"{'group':<11} {'subset':<16} {'images':>10} {'annotations':>13}  status"
    print(header)
    print("-" * len(header))
    total_anns = 0
    for r in rows:
        if r.get("missing"):
            status = "MISSING"
            n_img = n_ann = "-"
        elif r.get("error"):
            status = f"ERROR: {r['error']}"
            n_img = n_ann = "-"
        else:
            status = "ok"
            n_img, n_ann = r["n_images"], r["n_annotations"]
            total_anns += n_ann
        print(f"{r['group']:<11} {r['subset']:<16} {n_img!s:>10} {n_ann!s:>13}  {status}")
    print("-" * len(header))
    print(f"Total annotations (converted subsets only): {total_anns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())