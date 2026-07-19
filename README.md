# 2D Hand Keypoint Estimation — MAE vs DINOv3 backbones

Reproduction package for thesis experiment analyzing the performance of 2d keypoint 
detectors against the COCO wholebody hand dataset. Adds 2d keypoint data from [HaMeR](https://github.com/geopavlakos/hamer), [SynthMoCap](https://github.com/microsoft/SynthMoCap), and [CoCo Wholebody](https://github.com/jin-s13/COCO-WholeBody) and also compares the performance of DINOv3 and MAE backbone for this task.

This repository was built using the repository from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose/blob/main/mmpose/datasets/datasets/hand/hand_coco_wholebody_dataset.py),
using backbones from MAE, [DINOv2](https://github.com/facebookresearch/dinov2), and [DINOv3](https://github.com/facebookresearch/dinov3).The instructions below will build the environment and fetch the data. Proceed after cloning repository into your machine.

## 1. Environment Setup
A setup script has been prepared to ensure reproducibility. Doing this setup requires conda be installed on the machine. Do not upgrade packages, as some surgery had to be done to integrate DINOv3 (a pytorch v2 repository) into the ViTPose repository.

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
│   ├── train2017/  val2017/
│   └── annotations/coco_wholebody_{train,val}_v1.0.json
└── hamer_evaluation_data/
```
`REPO/data` is the expected location for this data for training, but you can easily place the data in another directory if desired.
If this is done, it is simplest to keep this other directory in the same format, and symlink REPO/data with your other directory. For each extraction script below, you may call it using
`DATA_ROOT=<DATA_ROOT> bash script/...` to overwrite the default `<REPO>/data`. We will refer to the directory you place the data as `<DATA_DIR>`.

### 2a. HaMeR training data (200 GB after extraction, around ~400 needed for full extraction process)
`fetch_hamer_data.sh` has been adapted from `fetch_training_data.sh` from the HaMeR repository.

```bash
bash scripts/fetch_hamer_data.sh      # gdown from Google Drive; Dropbox URLs in-script as fallback
bash scripts/extract_hamer_data.sh    # outer .tar.gz then nested dataset tar shards
WORKERS=$(nproc) bash scripts/convert_all_annotations.sh
```
Note that you may run `REMOVE=1 bash scrpts/...` in order to delete any tars after they have been downloaded or extracted, if you are concerned about running out of memory.

### 2b. SynthMoCap (SynthHand, ~8GB)

Downloader needs its own Python 3.10 env (separate from the training env) and
**your own logins for https://amass.is.tue.mpg.de/ and https://mano.is.tue.mpg.de/**. Also needs
system `wget`. You may find additional instructions at the [SynthMoCap Repository](https://github.com/microsoft/SynthMoCap.git).

```bash
conda create -n synthmocap python=3.10 pip -y && conda activate synthmocap 
git clone https://github.com/microsoft/SynthMoCap.git && cd SynthMoCap # Not used after data extraction so does not need to be done in <REPO>
pip install -r requirements.txt
python download_data.py --dataset hand --output-dir <DATA_DIR>/synthmocap/
cd <REPO>
```

Convert SynthMoCap (done back in the training env). SynthMoCap stores landmarks in MANO+tips
order; the `--reorder` mapping below converts to COCO hand order. This is the only dataset that needs reordering:

```bash
conda activate 2dKeypointHand
python scripts/convert_synthmocap.py \
    --input-dir data/synthmocap/synth_hand \
    --reorder "0,13,14,15,20,1,2,3,16,4,5,6,17,10,11,12,19,7,8,9,18" \
    --workers 8
```

(Mapping derivation: SynthMoCap order is wrist; index/middle/pinky/ring/thumb
3-joint chains; then 5 tips in the same finger order — from `LDMK_CONN` in
SynthMoCap's `visualize_data.py`. Verify visually with
`scripts/visualize_predictions.py` on a few samples: a wrong mapping shows up
instantly as crossed fingers.)

### 2c. COCO-WholeBody (~41GB)

```bash
bash scripts/fetch_coco.sh                      # val2017, train2017, and annotations
```

These images are downloaded using the URLS [here](https://cocodataset.org/#download), and links for annotations were found [here](https://github.com/jin-s13/COCO-WholeBody).

## 3. Pretrained Backbones
These will go into the `<REPO>/pretrained/` directory. These form one of the axes on which the experiments in this repository are done.

### 3a. MAE Backbone
This is the same backbone that ViTPose was trained with. Download links are found at the [MAE Repository](https://github.com/facebookresearch/mae), for example:
```bash
wget -c https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -P pretrained/
wget -c https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth -P pretrained/
wget -c https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth -P pretrained/
```

### 3b. DINOv3 Backbone
Acquiring this backbone requires going through links on Meta's [DINOv3 repository](https://github.com/facebookresearch/dinov3). You must request access by clicking 
into one of the models, and you will be given download links in a follow-up email. Note that these links likely need to be put into quotation marks (' ') in order to be parsed correctly, and -O should be used to ensure that the model is named in accordance with the model configuration's expectations, i.e. 
```bash
wget 'https:/...dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth...' -O pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```
Configuration exists for `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (ViT-S/16 distilled), `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (ViT-B/16 distilled), `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (ViT-L/16 distilled), and `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` (ViT-H+/16 distilled).

## 4. Train
Training is done through the infrastructure built for ViTPose, and you may see that repository for further instructions on running training. Our training config exists in 
`<REPO>/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset`, and example of running training using the DINOv3 and MAE backbone, all data, and on a single machine is 
```bash
bash tools/dist_train.sh configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset/DINOv3_base_hand_multidataset.py <NUM_GPUS> --cfg-options model.pretrained=pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
bash tools/dist_train.sh configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/multi_dataset/ViTPose_base_hand_multidataset.py <NUM_GPUS> --cfg-options model.pretrained=pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## 5. Eval
Eval is done using the infrastructure from [HaMeR](https://github.com/geopavlakos/hamer). You may download the [evaluation data](https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?e=1&rlkey=nb4x10uc8mj2qlfq934t5mdlh) and extract it to the data directory, i.e.
```bash
wget -c 'https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh&dl=1'      -O hamer_evaluation_data.tar.gz
tar -xzf hamer_evaluation_data.tar.gz -C data/
rm hamer_evaluation_data.tar.gz
```
Then, you may run an evaluation against using HaMeR's PCK calculator with 
```bash
python scripts/eval_PCK.py 
```
