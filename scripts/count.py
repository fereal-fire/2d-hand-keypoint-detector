import json
from pathlib import Path

data_root = "data"

train = [
    dict(
        type="FreihandHamerHandDataset",
        ann_file=f"{data_root}/hamer/freihand-train/annotations/coco_annotations.json",
    ),
    dict(
        type="SynthMocapHandDataset",
        ann_file=f"{data_root}/synthmocap/synth_hand/annotations/coco_annotations.json",
    ),
    dict(
        type="HandCocoWholeBodyDataset",
        ann_file=f"{data_root}/coco/annotations/coco_wholebody_train_v1.0.json",
    ),
    dict(
        type="Dexs0HandDataset",
        ann_file=f"{data_root}/hamer/dexs0-train/annotations/coco_annotations.json",
    ),
    dict(
        type="H2O3DHandDataset",
        ann_file=f"{data_root}/hamer/h2o3d-train/annotations/coco_annotations.json",
    ),
    dict(
        type="HO3DHandDataset",
        ann_file=f"{data_root}/hamer/ho3d-train/annotations/coco_annotations.json",
    ),
    dict(
        type="HalpeHandDataset",
        ann_file=f"{data_root}/hamer/halpe-train/annotations/coco_annotations.json",
    ),
    dict(
        type="InterHand26MDataset",
        ann_file=f"{data_root}/hamer/interhand26m-train/annotations/coco_annotations.json",
    ),
    dict(
        type="MPIINZSLHandDataset",
        ann_file=f"{data_root}/hamer/mpiinzsl-train/annotations/coco_annotations.json",
    ),
    dict(
        type="MTCHandDataset",
        ann_file=f"{data_root}/hamer/mtc-train/annotations/coco_annotations.json",
    ),
    dict(
        type="RHDHandDataset",
        ann_file=f"{data_root}/hamer/rhd-train/annotations/coco_annotations.json",
    ),
]


def is_valid(value):
    """Treat a hand as valid when its scalar flag is true or its list has any true value."""
    if isinstance(value, (list, tuple)):
        return any(value)
    return bool(value)


total_left_hands = 0
total_right_hands = 0
total_hands = 0

for dataset in train:
    ann_path = Path(dataset["ann_file"])

    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as file:
        annotation_data = json.load(file)

    annotations = annotation_data.get("annotations", [])

    left_hands = sum(
        is_valid(ann.get("lefthand_valid", False))
        for ann in annotations
    )

    right_hands = sum(
        is_valid(ann.get("righthand_valid", False))
        for ann in annotations
    )

    hand_count = left_hands + right_hands
    annotations_with_hands = sum(
        is_valid(ann.get("lefthand_valid", False))
        or is_valid(ann.get("righthand_valid", False))
        for ann in annotations
    )

    total_left_hands += left_hands
    total_right_hands += right_hands
    total_hands += hand_count

    print(
        f"{dataset.get('type', 'Unknown')}: "
        f"{hand_count:,} hands "
        f"({left_hands:,} left, {right_hands:,} right) across "
        f"{annotations_with_hands:,} annotations / "
        f"{len(annotations):,} total annotations"
    )

print()
print(f"Total left hands:  {total_left_hands:,}")
print(f"Total right hands: {total_right_hands:,}")
print(f"Total hands:       {total_hands:,}")