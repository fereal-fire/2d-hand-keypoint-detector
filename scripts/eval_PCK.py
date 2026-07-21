#!/usr/bin/env python3
"""Evaluate one model on HInt dataset/splits with HaMeR-style PCK. Self-contained:
no per-split test configs and no hamer import needed.

For each dataset/split it takes the model config's own data.test dict and
overrides ann_file/img_prefix to point at that split's converted COCO JSON and
image dir, runs MMPose inference (model is built and loaded once), then scores
predictions against the HInt GT npz with an inlined copy of HaMeR's EvaluatorPCK.

Expected data layout (override roots via flags):
  <npz-dir>/TEST_{name}_img_{split}.npz                 # HaMeR eval GT
  <npz-dir>/annotations/hint_TEST_{name}_img_{split}.json  # from convert_HInt_npz.py
  data/HInt_annotation_partial/TEST_{name}_img/         # eval images
<npz-dir> defaults to the first of data/hamer_evaluation_data,
data/hamer/hamer_evaluation_data that exists.

Usage:
    python scripts/eval_pck.py \
        configs/.../DINOv3_base_hand_multidataset.py \
        work_dirs/DINOv3_base_hand_multidataset/epoch_30.pth \
        --datasets NEWDAYS-TEST-ALL NEWDAYS-TEST-VIS NEWDAYS-TEST-OCC \
                   EPICK-TEST-ALL EPICK-TEST-VIS EPICK-TEST-OCC

Dataset identifiers follow HaMeR's eval.py convention: <NAME>-TEST-<SPLIT>
with NAME in {NEWDAYS, EPICK, EGO4D} and SPLIT in {ALL, VIS, OCC}.
A single comma-separated string is also accepted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import patch_version ## Custom

import argparse
import copy
import datetime
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmpose.apis import single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


# ---- HaMeR's EvaluatorPCK (inlined from hamer.utils.pose_utils) -------------

class EvaluatorPCK:
    def __init__(self, thresholds: List = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        self.thresholds = thresholds
        self.pred_kp_2d = []
        self.gt_kp_2d = []
        self.gt_conf_2d = []
        self.scale = []
        self.counter = 0

    def get_metrics_dict(self) -> Dict:
        pcks = self.compute_pcks()
        metrics = {}
        for thr, (acc, avg_acc, cnt) in zip(self.thresholds, pcks):
            metrics.update({f'kp{i}_pck_{thr}': float(a) for i, a in enumerate(acc) if a >= 0})
            metrics.update({f'kpAvg_pck_{thr}': float(avg_acc)})
        return metrics

    def compute_pcks(self):
        pred_kp_2d = np.concatenate(self.pred_kp_2d, axis=0)
        gt_kp_2d = np.concatenate(self.gt_kp_2d, axis=0)
        gt_conf_2d = np.concatenate(self.gt_conf_2d, axis=0)
        scale = np.concatenate(self.scale, axis=0)
        assert pred_kp_2d.shape == gt_kp_2d.shape
        assert pred_kp_2d[..., 0].shape == gt_conf_2d.shape
        assert pred_kp_2d.shape[1] == 1  # num_samples
        assert scale.shape[0] == gt_conf_2d.shape[0]

        return [
            self.keypoint_pck_accuracy(
                pred_kp_2d[:, 0, :, :],
                gt_kp_2d[:, 0, :, :],
                gt_conf_2d[:, 0, :] > 0.5,
                thr=thr,
                scale=scale[:, None],
            )
            for thr in self.thresholds
        ]

    def keypoint_pck_accuracy(self, pred, gt, conf, thr, scale):
        dist = np.sqrt(np.sum((pred - gt) ** 2, axis=2))
        all_joints = conf > 0.5
        correct_joints = np.logical_and(dist <= scale * thr, all_joints)
        pck = correct_joints.sum(axis=0) / all_joints.sum(axis=0)
        return pck, pck.mean(), pck.shape[0]

    def __call__(self, output: Dict, batch: Dict, opt_output: Optional[Dict] = None):
        pred_keypoints_2d = output['pred_keypoints_2d'].detach()
        num_samples = 1
        batch_size = pred_keypoints_2d.shape[0]

        right = batch['right'].detach()
        pred_keypoints_2d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_2d[:, :, 0]
        box_size = batch['box_size'].detach()
        box_center = batch['box_center'].detach()
        bbox_expand_factor = batch['bbox_expand_factor'].detach()
        scale = box_size / bbox_expand_factor
        pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None]
        pred_keypoints_2d = pred_keypoints_2d[:, None, :, :]
        gt_keypoints_2d = batch['orig_keypoints_2d'][:, None, :, :].repeat(1, num_samples, 1, 1)

        self.pred_kp_2d.append(pred_keypoints_2d[:, :, :, :2].detach().cpu().numpy())
        self.gt_conf_2d.append(gt_keypoints_2d[:, :, :, -1].detach().cpu().numpy())
        self.gt_kp_2d.append(gt_keypoints_2d[:, :, :, :2].detach().cpu().numpy())
        self.scale.append(scale.detach().cpu().numpy())

        self.counter += batch_size


# ---- scoring (from eval_vitpose_hint.py) ------------------------------------

def score_hamer_pck(outputs, npz_path: str, thresholds):
    d = np.load(npz_path, allow_pickle=True)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in d['imgname']]
    center = d['center'].astype(np.float64)                        # (N,2) px
    scale = d['scale'].reshape(len(names), -1).astype(np.float64)  # (N,2) px
    right = d['right'].astype(np.float64)                          # (N,)
    gt = d['hand_keypoints_2d'].astype(np.float64)                 # (N,21,3)
    N = len(names)

    # Box metadata exactly as HaMeR's ImageDataset HInt branch (rescale_factor=2)
    box_size = 2.0 * scale.max(axis=1)
    bbox_expand = np.full(N, 2.0)

    preds_by_name = {}
    for rec in outputs:
        for kp, path in zip(rec['preds'], rec['image_paths']):
            preds_by_name[os.path.basename(path)] = np.asarray(kp, dtype=np.float64)[:, :2]
    print(f'Collected {len(preds_by_name)} predictions; npz has {N} samples')

    P = np.full((N, 21, 2), np.nan, dtype=np.float64)
    n_missing = 0
    for i, nm in enumerate(names):
        key = os.path.basename(nm)
        if key in preds_by_name:
            P[i] = preds_by_name[key]
        else:
            n_missing += 1
    if n_missing:
        print(f'WARNING: {n_missing}/{N} samples had no matching prediction (scored as misses)')

    # Invert HaMeR's normalization so EvaluatorPCK reconstructs P exactly
    pred_after = (P - center[:, None, :]) / box_size[:, None, None]
    pred_norm = pred_after.copy()
    pred_norm[:, :, 0] = (2 * right[:, None] - 1) * pred_after[:, :, 0]

    evaluator = EvaluatorPCK(thresholds=list(thresholds))
    chunk = 512
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        output = {'pred_keypoints_2d': torch.from_numpy(pred_norm[s:e]).float()}
        batch = {
            'right': torch.from_numpy(right[s:e]).float(),
            'box_size': torch.from_numpy(box_size[s:e]).float(),
            'box_center': torch.from_numpy(center[s:e]).float(),
            'bbox_expand_factor': torch.from_numpy(bbox_expand[s:e]).float(),
            'orig_keypoints_2d': torch.from_numpy(gt[s:e]).float(),
        }
        evaluator(output, batch)

    return evaluator.get_metrics_dict()


# ---- main --------------------------------------------------------------------

DEFAULT_DATASETS = [
    'NEWDAYS-TEST-ALL', 'NEWDAYS-TEST-VIS', 'NEWDAYS-TEST-OCC',
    'EPICK-TEST-ALL', 'EPICK-TEST-VIS', 'EPICK-TEST-OCC',
]

VALID_NAMES = {'NEWDAYS': 'newdays', 'EPICK': 'epick', 'EGO4D': 'ego4d'}
VALID_SPLITS = {'ALL': 'all', 'VIS': 'vis', 'OCC': 'occ'}


def parse_dataset_id(dataset_id: str):
    """'NEWDAYS-TEST-ALL' -> ('newdays', 'all'). HaMeR eval.py naming."""
    parts = dataset_id.upper().split('-')
    if len(parts) != 3 or parts[1] != 'TEST' or \
            parts[0] not in VALID_NAMES or parts[2] not in VALID_SPLITS:
        raise ValueError(
            f"Bad dataset id '{dataset_id}': expected <NAME>-TEST-<SPLIT> with "
            f"NAME in {sorted(VALID_NAMES)} and SPLIT in {sorted(VALID_SPLITS)}")
    return VALID_NAMES[parts[0]], VALID_SPLITS[parts[2]]

def main():
    ap = argparse.ArgumentParser(
        description='HaMeR-PCK eval of one model over HInt dataset/splits (no test configs needed)')
    ap.add_argument('model_config', help='config providing model + data.test template')
    ap.add_argument('checkpoint', help='trained checkpoint .pth')
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                    help="dataset ids like NEWDAYS-TEST-ALL (HaMeR eval.py style); "
                         "space- or comma-separated")
    ap.add_argument('--data-root', default='data/hamer')
    ap.add_argument('--img-root', default=None, help='default: <data-root>/HInt_annotation_partial')
    ap.add_argument('--npz-dir', default=None,
                    help='default: <data-root>/hamer_evaluation_data')
    ap.add_argument('--ann-dir', default=None, help='default: <npz-dir>/annotations')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.15])
    ap.add_argument('--samples-per-gpu', type=int, default=32)
    ap.add_argument('--pred-dir', default='model_predictions', help='where to save prediction pkls')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    img_root = Path(args.img_root) if args.img_root else data_root / 'HInt_annotation_partial'
    npz_dir = Path(args.npz_dir) if args.npz_dir else data_root / 'hamer_evaluation_data'
    ann_dir = Path(args.ann_dir) if args.ann_dir else npz_dir / 'annotations'

    cfg = Config.fromfile(args.model_config)
    cfg.model.pretrained = None 
    
    model = build_posenet(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    run = Path(args.checkpoint).stem
    model_name = Path(args.model_config).stem

    # accept both space-separated args and a single comma-separated string
    dataset_ids = [d for arg in args.datasets for d in arg.split(',') if d]
    combos = [parse_dataset_id(d) for d in dataset_ids]

    results = []   # (name, split, metrics_dict)
    failures = []

    for name, split in combos:
        tag = f'{name}_{split}'
        img_prefix = img_root / f'TEST_{name}_img'
        ann_file = ann_dir / f'hint_TEST_{name}_img_{split}.json'
        npz = npz_dir / f'TEST_{name}_img_{split}.npz'

        missing = [p for p in (ann_file, img_prefix, npz) if not p.exists()]
        if missing:
            print(f'[SKIP] {tag}: missing ' + ', '.join(str(m) for m in missing))
            failures.append(tag)
            continue

        print(f'\n===== {tag} =====')

        # data.test template from the model config, retargeted at this split
        test_dict = copy.deepcopy(cfg.data.test)
        test_dict['ann_file'] = str(ann_file)
        test_dict['img_prefix'] = str(img_prefix) + '/'

        dataset = build_dataset(test_dict, dict(test_mode=True))
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
            dist=False,
            shuffle=False,
        )

        try:
            outputs = single_gpu_test(model, dataloader)
        except Exception as e:  # noqa: BLE001
            print(f'[FAIL] {tag}: inference failed: {e}')
            failures.append(tag)
            continue

        preds_path = Path(args.pred_dir) / tag / f'{run}.pkl'
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        with preds_path.open('wb') as f:
            pickle.dump(outputs, f)

        md = score_hamer_pck(outputs, str(npz), args.thresholds)
        print(f'=== HaMeR PCK for {npz.name} ===')
        for thr in args.thresholds:
            key = f'kpAvg_pck_{thr}'
            if key in md:
                print(f'  PCK@{thr}: {md[key]:.4f}')
        results.append((name, split, md))

    # ---- summary ----
    if results:
        print(f'\n===== Summary: {model_name} {run} =====')
        header = f"{'dataset':10s} {'split':5s}" + ''.join(f'  PCK@{t:<5g}' for t in args.thresholds)
        print(header)
        for name, split, md in results:
            row = f'{name:10s} {split:5s}'
            row += ''.join(f"  {md.get(f'kpAvg_pck_{t}', float('nan')):8.4f}" for t in args.thresholds)
            print(row)

    if failures:
        print(f'\nFailed/skipped: {", ".join(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
