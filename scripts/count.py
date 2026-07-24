
import json
from pathlib import Path

data_root = "data"
 # dict(
        # type='HalpeHandDataset',
        # ann_file=f'/mnt/hamer_datasets/halpe/annotations/annotations_train'),
        # dict(
        # type='HalpeHandDataset',
        # ann_file=f'/mnt/hamer_datasets/halpe/annotations/annotations_train_subset.json'),
        # dict(
        # type='HalpeHandDataset',
        # ann_file=f'/mnt/hamer_datasets/halpe/annotations/annotations_train_subset_2.json'),
        # dict(
        # type='MPIINZSLHandDataset',
        # ann_file=f'/mnt/hamer_datasets/mpiinzsl/annotations/annotations_train'),
        # dict(
        # type='MPIINZSLHandDataset',
        # ann_file=f'/mnt/hamer_datasets/mpiinzsl/annotations/annotations_train_subset.json'),
        # dict(
        # type='MPIINZSLHandDataset',
        # ann_file=f'/mnt/hamer_datasets/mpiinzsl/annotations/annotations_train_subset_2.json'),
        # dict(
        # type='RHDHandDataset2',
        # ann_file=f'{data_root}/hamer/rhd-train/annotations/full_dataset_coco_annotations.json'),
        # dict(
        # type='RHDHandDataset3',
        # ann_file=f'/mnt/hamer_datasets/rhd/annotations/annotations_train'),
        # dict(
        # type='RHDHandDataset4',
        # ann_file=f'/mnt/hamer_datasets/rhd/annotations/annotations_train_subset.json'),
train =   [
        dict(
        type='FreihandHamerHandDataset',
        ann_file=f'{data_root}/hamer/freihand-train/annotations/coco_annotations.json'),
        dict(
        type='SynthMocapHandDataset',
        ann_file=f'{data_root}/synthmocap/synth_hand/annotations/coco_annotations.json'),
        dict(
        type='HandCocoWholeBodyDataset',
        ann_file=f'{data_root}/coco/annotations/coco_wholebody_train_v1.0.json'),
        dict(
        type='Dexs0HandDataset',
        ann_file=f'{data_root}/hamer/dexs0-train/annotations/coco_annotations.json'),
        dict(
        type='H2O3DHandDataset',
        ann_file=f'{data_root}/hamer/h2o3d-train/annotations/coco_annotations.json'),
        dict(
        type='HO3DHandDataset',
        ann_file=f'{data_root}/hamer/ho3d-train/annotations/coco_annotations.json'),
        dict(
        type='HalpeHandDataset',
        ann_file=f'{data_root}/hamer/halpe-train/annotations/coco_annotations.json'),
        dict(
        type='InterHand26MDataset',
        ann_file=f'{data_root}/hamer/interhand26m-train/annotations/coco_annotations.json'),
        dict(
        type='MPIINZSLHandDataset',
        ann_file=f'{data_root}/hamer/mpiinzsl-train/annotations/coco_annotations.json'),
        dict(
        type='MTCHandDataset',
        ann_file=f'{data_root}/hamer/mtc-train/annotations/coco_annotations.json'),
        dict(
        type='RHDHandDataset',
        ann_file=f'{data_root}/hamer/rhd-train/annotations/coco_annotations.json'),
        ]

def is_valid(value):
    """Handle scalar flags or lists of per-keypoint validity values."""
    if isinstance(value, (list, tuple)):
        return any(value)
    return bool(value)


n_total = 0

for dataset in train:
    ann_path = Path(dataset["ann_file"])

    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as file:
        annotation_data = json.load(file)

    annotations = annotation_data.get("annotations", [])

    valid_annotations = [
        ann
        for ann in annotations
        if is_valid(ann.get("lefthand_valid", False))
        or is_valid(ann.get("righthand_valid", False))
    ]

    n_ann = len(valid_annotations)
    n_total += n_ann

    print(
        f"{dataset.get('type', 'Unknown')}: "
        f"{n_ann:,} valid / {len(annotations):,} total"
    )

print(f"Total valid annotations: {n_total:,}")