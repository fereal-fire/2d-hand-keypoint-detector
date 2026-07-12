# 2D Hand Keypoint Estimation — MAE vs DINOv3 backbones

Reproduction package for the thesis experiment analyzing the performance of 2d keypoint 
detectors against the COCO wholebody hand dataset. Adds 2d keypoint data from [HaMeR](https://github.com/geopavlakos/hamer) and 
synthmocap, and also compares the performance of DINOv3 and MAE backbone for this task.

This repository was built using the repository from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose/blob/main/mmpose/datasets/datasets/hand/hand_coco_wholebody_dataset.py),
using backbones from MAE, [DINOv2](https://github.com/facebookresearch/dinov2), and [DINOv3](https://github.com/facebookresearch/dinov3).The instructions below will build the environment and fetch the data. Proceed after cloning repository into your machine.

## 1. Environment Setup
A setup script has been prepared to ensure reproducibility. Doing this setup requires conda be installed on the machine.

To set up the environment:
```bash
bash scripts/setup_env.sh        # creates conda env "2dKeypointHand" and verifies imports
conda activate 2dKeypointHand
```
The script creates the env `2dKeypointHand` from `environment.yml`, adds third party packages (dinov2 and dinov3) to path,
and proceed through ViTPose's setup.

## 2. Data

Expected layout under `data/` after the steps below:

```
data/
├── hamer/                          # HaMeR training data (freihand-train/, h2o3d-train/, ...)
│   └── <dataset>/annotations/coco_annotations.json   # produced by converter
├── SynthMoCap/synth_hand/                     # SynthMoCap SynthHand
│   └── annotations/coco_synthmocap_annotation.json
├── coco/                           # COCO images + WholeBody hand annotations
    ├── train2017/  val2017/
    └── annotations/coco_wholebody_{train,val}_v1.0.json
```

`REPO/data` is the expected location for this data for training, but you can easily place the data in the follow instructions in another directory if desired.
If this is done, it is simplest to keep this other directory in the same format, and symlink REPO/data with your other directory. For each extraction script below, you may call it using
`DATA_ROOT=<DATA_ROOT> bash script/...` to overwrite the default `<REPO>/data`.

### 2a. HaMeR training data (~300 GB)
`fetch_hamer_data.sh` has been adapted from `fetch_training_data.sh` from the HaMeR repository.

```bash
bash scripts/fetch_hamer_data.sh      # gdown from Google Drive; Dropbox URLs in-script as fallback
bash scripts/extract_hamer_data.sh    # outer .tar.gz then nested dataset tar shards
WORKERS=$(nproc) bash scripts/convert_all_annotations.sh
```
Note that you may run `REMOVE=1 bash scrpts/...` in order to delete any tars after they have been deleted or extracted, if you are concerned about running out of memory.

### 2b. SynthMoCap (SynthHand, ~7GB)

Downloader needs its own Python 3.10 env (separate from the training env) and
**your own logins for https://amass.is.tue.mpg.de/ and https://mano.is.tue.mpg.de/**. Also needs
system `wget`. You may find additional instructions at the [SynthMoCap Repository](https://github.com/microsoft/SynthMoCap.git).

```bash
conda create -n synthmocap python=3.10 pip -y && conda activate synthmocap # Not used after data extraction so does not need to be done in <REPO>
git clone https://github.com/microsoft/SynthMoCap.git && cd SynthMoCap
pip install -r requirements.txt
python download_data.py --dataset hand --output-dir <REPO>/data/   # -> data/synth_hand/
cd <REPO>
```

Convert (back in the training env). SynthMoCap stores landmarks in MANO+tips
order; the `--reorder` mapping below converts to COCO hand order. This is the only dataset that needs reordering:

```bash
conda activate 2dKeypointHand
python scripts/convert_synthmocap.py \
    --input-dir data/synth_hand \
    --reorder "0,13,14,15,20,1,2,3,16,4,5,6,17,10,11,12,19,7,8,9,18" \
    --workers 8
```

(Mapping derivation: SynthMoCap order is wrist; index/middle/pinky/ring/thumb
3-joint chains; then 5 tips in the same finger order — from `LDMK_CONN` in
SynthMoCap's `visualize_data.py`. Verify visually with
`scripts/visualize_predictions.py` on a few samples: a wrong mapping shows up
instantly as crossed fingers.)

### 2c. COCO-WholeBody (hand subset)

```bash
bash scripts/fetch_coco.sh                      # val2017, train2017, and annotations
```

These images are downloaded using the URLS [here](https://cocodataset.org/#download), and links for annotations were found [here](https://github.com/jin-s13/COCO-WholeBody).

## Pretrained Backbones


## 4. Train

ViTpose's repository can be consulted for more information on utilizing their pipeline. Our config for our models exist in 
`<REPO>/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset`. We can run using

```bash
# single GPU node:
bash tools/dist_train.sh configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset/<Model> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH> --seed 0

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
