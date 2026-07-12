# 2D Hand Keypoint Estimation — MAE vs DINOv3 backbones

Reproduction package for the thesis experiment analyzing the performance of 2d keypoint 
detectors against the COCO wholebody hand dataset. Adds 2d keypoint data from HaMeR and 
synthmocap, and also compares the performance of DINOv3 and MAE backbone for this task.

## Repo layout

| Path | Contents |
|---|---|
| `configs/` | MMPose training configs (MAE + DINOv3) |
| `mmpose/models/backbones/` | Custom `DINOv2`/`DINOv3` backbones (part of the MMPose fork) |
| `scripts/` | Data fetch/extract/convert, HaMeR-PCK scoring, visualization |
| `jobs/` | `train.sh` / `train.sbatch` (SLURM) / `eval.sh` launchers |
| `environment/` | `environment.yml` + pinned requirements |
| `results/` | Results table (`hint_eval_results.csv`) |
| `data/` | Not in git — created during setup (layout below) |
| `work_dirs/` | Not in git — MMPose training outputs |

## 1. 

```bash
bash scripts/setup_env.sh        # creates conda env "2dKeypointHand" and verifies imports
conda activate 2dKeypointHand
```

The script: creates the env from `environment/environment.yml`, installs the
mmcv-full 1.3.9 cu111 wheel, `pip install -e .` (this fork), then
`timm==0.4.9 einops orjson`, and registers `third_party/dinov2` on the
Python path via a `.pth` file in site-packages (the `DINOv2` backbone imports
`dinov2.models`; do **not** pip-install dinov2's own requirements — they pin
torch 2.0).

Requires: conda, `libGL.so.1` (`apt install libgl1 libglib2.0-0`), a CUDA 11.x
GPU for training (the setup check warns but passes on CPU-only nodes).

**TODO(alex):** commit `pip freeze > environment/requirements-exact.txt` from
the VM + record Python/CUDA/driver versions here.

### HaMeR for scoring (evaluation only)

`scripts/eval_vitpose_hint.py` imports `hamer.utils.pose_utils.EvaluatorPCK`
so reported PCK is computed by HaMeR's own code, not a reimplementation:

```bash
git clone https://github.com/geopavlakos/hamer.git ~/hamer
export PYTHONPATH=~/hamer:$PYTHONPATH    # no need for hamer's full install
```

## 2. Data

Expected layout under `data/` after the steps below:

```
data/
├── hamer/                          # HaMeR training data (freihand-train/, h2o3d-train/, ...)
│   └── <dataset>/annotations/coco_annotations.json   # produced by converter
├── synth_hand/                     # SynthMoCap SynthHand
│   └── annotations/coco_synthmocap_annotation.json
├── coco/                           # COCO images + WholeBody hand annotations
│   ├── train2017/  val2017/
│   └── annotations/coco_wholebody_{train,val}_v1.0.json
├── HInt_annotation_partial/        # eval images (NewDays, EPIC-Kitchens VISOR)
├── hamer_evaluation_data/          # HInt eval GT npz files
└── annotations/                    # HInt eval COCO JSONs (from converter)
```

### 2a. HaMeR training data (~300GB raw; needs ~2x in free disk)

```bash
bash scripts/fetch_hamer_data.sh      # gdown from Google Drive; Dropbox URLs in-script as fallback
bash scripts/extract_hamer_data.sh    # outer .tar.gz then nested dataset tar shards
WORKERS=$(nproc) bash scripts/convert_all_annotations.sh
```

Gotchas: Google Drive quota errors are common — verify each archive with
`gzip -t` (the fetch script does this) and fall back to the Dropbox links;
conversion is disk-I/O bound, more workers won't help much on HDD.
No `--reorder` is needed for HaMeR (keypoints are already in COCO hand order).

### 2b. SynthMoCap (SynthHand, ~7GB)

Downloader needs its own Python 3.10 env (separate from the training env) and
**your own logins for https://amass.is.tue.mpg.de/ and https://mano.is.tue.mpg.de/**
(pose data is spliced in at download time, not redistributed). Also needs
system `wget` with TLSv1.2.

```bash
conda create -n synthmocap python=3.10 pip -y && conda activate synthmocap
git clone https://github.com/microsoft/SynthMoCap.git && cd SynthMoCap
pip install -r requirements.txt
python download_data.py --dataset hand --output-dir <REPO>/data/   # -> data/synth_hand/
```

Convert (back in the training env). SynthMoCap stores landmarks in MANO+tips
order; the `--reorder` mapping below converts to COCO hand order and **must
match training** — this is the only dataset that needs reordering:

```bash
conda activate 2dKeypointHand
python scripts/convert_synthmocap.py \
    --input-dir data/synth_hand \
    --reorder "0,13,14,15,20,1,2,3,16,4,5,6,17,10,11,12,19,7,8,9,18" \
    --hand-side left \
    --workers 8
```

(Mapping derivation: SynthMoCap order is wrist; index/middle/pinky/ring/thumb
3-joint chains; then 5 tips in the same finger order — from `LDMK_CONN` in
SynthMoCap's `visualize_data.py`. Verify visually with
`scripts/visualize_predictions.py` on a few samples: a wrong mapping shows up
instantly as crossed fingers.)

### 2c. COCO-WholeBody (hand subset)

```bash
bash scripts/fetch_coco.sh                      # val2017 + train2017 + WholeBody JSONs
WITH_TRAIN_IMAGES=0 bash scripts/fetch_coco.sh  # eval-only variant, skips the 18GB train zip
```

Images come straight from images.cocodataset.org (use `aria2c -x16` on the
same URLs if wget is slow); the WholeBody hand-annotation JSONs come from the
Google Drive links in https://github.com/jin-s13/COCO-WholeBody (the script
knows the file IDs and detects Drive-quota HTML masquerading as JSON).

### 2d. HInt evaluation data

```bash
wget https://fouheylab.eecs.umich.edu/~dandans/projects/hamer/HInt_annotation_partial.zip
unzip HInt_annotation_partial.zip -d data/
```

Ego4D frames are excluded by license — see https://github.com/ddshan/hint.
The `hamer_evaluation_data/` npz files come with HaMeR's evaluation release
(https://github.com/geopavlakos/hamer).

Convert an eval npz to a COCO JSON for the test dataloader (once per
name/split):

```bash
python scripts/convert_hint_npz_to_coco.py \
    --npz data/hamer_evaluation_data/TEST_newdays_img_all.npz \
    --img-dir data/HInt_annotation_partial/TEST_newdays_img/ \
    --output-json data/annotations/hint_newdays_all.json
```

## 3. Pretrained backbones

- **MAE ViT** — **TODO(alex): exact checkpoint + URL (or config auto-download).**
- **DINOv3** — ViT-B/16 (embed_dim 768, depth 12, 256×256 input; see
  `configs/`). Request access and download from
  https://github.com/facebookresearch/dinov3.
  **TODO(alex): document the weight-conversion step** (configs set
  `pretrained=None`; the converted checkpoint is loaded via
  <mechanism + exact command here>).

## 4. Train

Run from the repo root (configs use the relative `data_root = 'data/...'`):

```bash
# single GPU node:
./jobs/train.sh configs/<mae_config>.py
./jobs/train.sh configs/<dinov3_config>.py

# or SLURM:
sbatch jobs/train.sbatch configs/<dinov3_config>.py
```

Checkpoints land in `work_dirs/<config-name>/`. **TODO(alex): confirm
training schedule and which datasets each thesis run trained on** (the
configs are ground truth). To test hypothesis (a), also try the DINOv3
config with a larger batch / more data.

## 5. Evaluate on HInt (HaMeR PCK)

One command per dataset/split (name = `newdays|epick|ego4d`,
split = `all|vis|occ`):

```bash
./jobs/eval.sh configs/<dinov3_config>.py \
    work_dirs/<dinov3_config>/epoch_30.pth newdays all
```

This runs MMPose inference (`tools/test.py --out preds.pkl`, predictions in
original image pixels) and scores the pkl with HaMeR's `EvaluatorPCK`,
printing PCK@0.05 / 0.10 / 0.15. Record numbers in
`results/hint_eval_results.csv`.

Notes baked into the metric (full derivation in the script docstrings):
a joint counts only if GT confidence > 0.5 (this is what distinguishes the
`all`/`vis`/`occ` npz files); the PCK normalization scale per sample is
`npz['scale'].max()` in pixels; handedness comes from `npz['right']` (GT
annotation — the model never predicts it).

## 6. Visualize (optional)

`scripts/run_visualizations.sh` renders side-by-side [GT | prediction] panels
(plus in-depth and error/confidence variants) for every pkl under
`model_predictions/<name>_<split>/`, into `model_visualizations/`:

```bash
# run from a directory containing model_predictions/, hamer_evaluation_data/
# and HInt_annotation_partial/ (paths are relative to the CWD):
./scripts/run_visualizations.sh          # DRY_RUN=1 to preview commands
```

## Smoke test (recommended before a full run)

Every stage has a cheap path: SynthMoCap `--single_chunk` (300MB),
`WITH_TRAIN_IMAGES=0` for COCO, converter `--size 500` (HaMeR) /
`--glob 'metadata_00000*'` (SynthMoCap), and a 1–2 epoch training run
(`total_epochs` override) just to prove the pipeline executes end to end.

## Expected results

**TODO(alex):** fill `results/hint_eval_results.csv` with the thesis MAE and
DINOv3 numbers (after the verification test run) so cluster runs can be
checked against them.

## Contact

Alex Wilcox — alexwilcox06@gmail.com
