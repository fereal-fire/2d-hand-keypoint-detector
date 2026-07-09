_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/synthmocap_hand.py'
]
evaluation = dict(interval=1, metric=['PCK', 'AUC', 'EPE'], key_indicator='AUC')
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=90, scale_factor=0.3),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]

test_pipeline = val_pipeline

data_root = '/mnt/hamer_datasets'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=[
        dict(
        type='FreihandHamerHandDataset',
        ann_file=f'{data_root}/freihand/annotations/annotations_train_subset.json',
        img_prefix=f'{data_root}/freihand/freihand-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='SynthMocapHandDataset',
        ann_file=f'/mnt/coco/synthmocap/annotations/synthmocap_train_subset.json',
        img_prefix=f'/mnt/coco/synthmocap/synth_hand/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='HandCocoWholeBodyDataset',
        ann_file=f'/mnt/coco/coco/annotations/coco_wholebody_train_v1.0.json',
        img_prefix=f'/mnt/coco/coco/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='Dexs0HandDataset',
        ann_file=f'{data_root}/dex/annotations/annotations_train_subset.json',
        img_prefix=f'{data_root}/dex/dexs0-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='H2O3DHandDataset',
        ann_file=f'{data_root}/h2o3d/annotations/annotations_train_subset.json',
        img_prefix=f'{data_root}/h2o3d/h2o3d-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='HO3DHandDataset',
        ann_file=f'{data_root}/ho3d/annotations/annotations_train_subset.json',
        img_prefix=f'{data_root}/ho3d/ho3d-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='HalpeHandDataset',
        ann_file=f'{data_root}/halpe/annotations/annotations_train_subset_2.json',
        img_prefix=f'{data_root}/halpe/halpe-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='InterHand26MDataset',
        ann_file=f'{data_root}/interhand26m/annotations/annotations_train_subset.json',
        img_prefix=f'{data_root}/interhand26m/interhand26m-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='MPIINZSLHandDataset',
        ann_file=f'{data_root}/mpiinzsl/annotations/annotations_train_subset_2.json',
        img_prefix=f'{data_root}/mpiinzsl/mpiinzsl-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='MTCHandDataset',
        ann_file=f'{data_root}/mtc/annotations/annotations_train_subset.json',
        img_prefix=f'{data_root}/mtc/mtc-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        dict(
        type='RHDHandDataset',
        ann_file=f'{data_root}/rhd/annotations/annotations_train_subset.json',
        img_prefix=f'{data_root}/rhd/rhd-train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
        ],
    # val=dict(
    #     type='HandCocoWholeBodyDataset',
    #     ann_file=f'/mnt/coco/coco/annotations/coco_wholebody_val_v1.0.json',
    #     img_prefix=f'/mnt/coco/coco/val2017/',
    #     data_cfg=data_cfg,
    #     pipeline=val_pipeline,
    #     dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='FreihandHamerHandDataset',
        ann_file=f'/mnt/hamer_datasets/hamer_ego4d_coco/out_newdays_vis/annotations_val_subset_2.json',
        img_prefix=f'/mnt/hamer_datasets/HInt_annotation_partial/TEST_newdays_img/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
