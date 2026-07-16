#!/usr/bin/env python3
"""Compose model from one config + test set from another, run inference, score
with HaMeR's own EvaluatorPCK.

Steps:
  1. load MODEL_CONFIG and take its `model` (+ runtime) section
  2. replace its data.test with the `data.test` dict from TEST_CONFIG
  3. run MMPose inference (equivalent of tools/test.py) over that test set
  4. match predictions to the HInt GT npz and compute HaMeR PCK@thresholds

Requires the hamer repo on PYTHONPATH (for hamer.utils.pose_utils.EvaluatorPCK).

Usage:
    python scripts/eval_composed.py \
        configs/.../DINOv3_base_hand_multidataset.py \
        work_dirs/DINOv3_base_hand_multidataset/epoch_30.pth \
        configs/.../hint_newdays_all_test.py \
        --npz data/hamer_evaluation_data/TEST_newdays_img_all.npz \
        --out model_predictions/newdays_all/epoch_30.pkl
"""
from typing import Dict, List, Optional
import argparse
import os
import pickle

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmpose.apis import single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet

##--------------------------------
class EvaluatorPCK:
    def __init__(self, thresholds: List = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],):
        """
        Class used for evaluating trained models on different 3D pose datasets.
        Args:
            thresholds [List]: List of PCK thresholds to evaluate.
            metrics [List]: List of evaluation metrics to record.
        """
        self.thresholds = thresholds
        self.pred_kp_2d = []
        self.gt_kp_2d = []
        self.gt_conf_2d = []
        self.scale = []
        self.counter = 0

    def log(self):
        """
        Print current evaluation metrics
        """
        if self.counter == 0:
            print('Evaluation has not started')
            return
        print(f'{self.counter} samples')
        metrics_dict = self.get_metrics_dict()
        for metric in metrics_dict:
            print(f'{metric}: {metrics_dict[metric]}')
        print('***')

    def get_metrics_dict(self) -> Dict:
        """
        Returns:
            Dict: Dictionary of evaluation metrics.
        """
        pcks = self.compute_pcks()
        metrics = {}
        for thr, (acc,avg_acc,cnt) in zip(self.thresholds, pcks):
            metrics.update({f'kp{i}_pck_{thr}': float(a) for i, a in enumerate(acc) if a>=0})
            metrics.update({f'kpAvg_pck_{thr}': float(avg_acc)})
        return metrics

    def compute_pcks(self):
        pred_kp_2d = np.concatenate(self.pred_kp_2d, axis=0)
        gt_kp_2d = np.concatenate(self.gt_kp_2d, axis=0)
        gt_conf_2d = np.concatenate(self.gt_conf_2d, axis=0)
        scale = np.concatenate(self.scale, axis=0)
        assert pred_kp_2d.shape == gt_kp_2d.shape
        assert pred_kp_2d[..., 0].shape == gt_conf_2d.shape
        assert pred_kp_2d.shape[1] == 1 # num_samples
        assert scale.shape[0] == gt_conf_2d.shape[0] # num_samples

        pcks = [
            self.keypoint_pck_accuracy(
                pred_kp_2d[:, 0, :, :],
                gt_kp_2d[:, 0, :, :],
                gt_conf_2d[:, 0, :]>0.5,
                thr=thr,
                scale = scale[:,None]
            )
            for thr in self.thresholds
        ]
        return pcks

    def keypoint_pck_accuracy(self, pred, gt, conf, thr, scale):
        dist = np.sqrt(np.sum((pred-gt)**2, axis=2))
        all_joints = conf>0.5
        correct_joints = np.logical_and(dist<=scale*thr, all_joints)
        pck = correct_joints.sum(axis=0)/all_joints.sum(axis=0)
        return pck, pck.mean(), pck.shape[0]

    def __call__(self, output: Dict, batch: Dict, opt_output: Optional[Dict] = None):
        """
        Evaluate current batch.
        Args:
            output (Dict): Regression output.
            batch (Dict): Dictionary containing images and their corresponding annotations.
            opt_output (Dict): Optimization output.
        """
        pred_keypoints_2d = output['pred_keypoints_2d'].detach()
        num_samples = 1
        batch_size = pred_keypoints_2d.shape[0]

        right = batch['right'].detach()
        pred_keypoints_2d[:,:,0] = (2*right[:,None]-1)*pred_keypoints_2d[:,:,0]
        box_size = batch['box_size'].detach()
        box_center = batch['box_center'].detach()
        bbox_expand_factor = batch['bbox_expand_factor'].detach()
        scale = box_size/bbox_expand_factor
        bbox_expand_factor = bbox_expand_factor[:,None,None,None]
        pred_keypoints_2d = pred_keypoints_2d*box_size[:,None,None]+box_center[:,None]
        pred_keypoints_2d = pred_keypoints_2d[:,None,:,:]
        gt_keypoints_2d = batch['orig_keypoints_2d'][:,None,:,:].repeat(1, num_samples, 1, 1)
        
        self.pred_kp_2d.append(pred_keypoints_2d[:, :, :, :2].detach().cpu().numpy())
        self.gt_conf_2d.append(gt_keypoints_2d[:, :, :, -1].detach().cpu().numpy())
        self.gt_kp_2d.append(gt_keypoints_2d[:, :, :, :2].detach().cpu().numpy())
        self.scale.append(scale.detach().cpu().numpy())

        self.counter += batch_size
##--------------------------------

def build_composed_cfg(model_config: str, test_config: str) -> Config:
    """Model/runtime from model_config; data.test (+ pipeline) from test_config."""
    cfg = Config.fromfile(model_config)
    tcfg = Config.fromfile(test_config)

    cfg.data.test = tcfg.data.test
    if 'test_dataloader' in tcfg.data:
        cfg.data.test_dataloader = tcfg.data.test_dataloader

    # We load trained weights explicitly; don't re-download the backbone init.
    cfg.model.pretrained = None
    return cfg


def run_inference(cfg: Config, checkpoint: str, samples_per_gpu: int):
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False,
    )
    model = build_posenet(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    return single_gpu_test(model, dataloader)


def score_hamer_pck(outputs, npz_path: str, thresholds):
    """Adapted from eval_vitpose_hint.py: EvaluatorPCK over HInt GT npz."""
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

    md = evaluator.get_metrics_dict()
    print(f'\n=== HaMeR PCK for {os.path.basename(npz_path)} ===')
    for thr in thresholds:
        key = f'kpAvg_pck_{thr}'
        if key in md:
            print(f'  PCK@{thr}: {md[key]:.4f}')
    return md


def main():
    ap = argparse.ArgumentParser(
        description='Compose model cfg + test cfg, run inference, score with HaMeR PCK')
    ap.add_argument('model_config', help='config providing model/runtime (e.g. training config)')
    ap.add_argument('checkpoint', help='trained checkpoint .pth')
    ap.add_argument('test_config', help='config providing data.test (dataset to run on)')
    ap.add_argument('--npz', required=True, help='HInt GT npz for scoring')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.05, 0.1, 0.15])
    ap.add_argument('--samples-per-gpu', type=int, default=32)
    ap.add_argument('--out', default=None, help='optional path to save predictions pkl')
    args = ap.parse_args()

    cfg = build_composed_cfg(args.model_config, args.test_config)
    print(f'Model from:   {args.model_config}')
    print(f'Test set from: {args.test_config}')
    print(f"Test ann_file: {cfg.data.test.get('ann_file')}")

    outputs = run_inference(cfg, args.checkpoint, args.samples_per_gpu)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'wb') as f:
            pickle.dump(outputs, f)
        print(f'Saved predictions to {args.out}')

    score_hamer_pck(outputs, args.npz, args.thresholds)


if __name__ == '__main__':
    main()