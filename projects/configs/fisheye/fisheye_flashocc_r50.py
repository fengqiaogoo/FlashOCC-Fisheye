_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-10.0, -10.0, -2.0, 10.0, 10.0, 2.0]

class_names = [
    'free', 'unknown', 'person', 'table', 'chair', 'floor', 'car'
]

data_config = {
    'cams': [
        'cam_front', 'cam_rear', 'cam_left', 'cam_right'
    ],
    'Ncams': 4,
    'input_size': (512, 640),
    'src_size': (512, 640),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-10, 10, 0.2],
    'y': [-10, 10, 0.2],
    'z': [-2, 2, 4],
    'depth': [1.0, 30.0, 0.5],
}

voxel_size = [0.2, 0.2, 0.2]

numC_Trans = 64

model = dict(
    type='FisheyeBEVDepthOCC',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepthFisheye',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 4 + numC_Trans * 2,
        out_channels=256,
        scale_factor=2,
        input_feature_index=(0, 1)),
    occ_head=dict(
        type='BEVOCCHead2D',
        in_dim=256,
        out_dim=256,
        Dz=20,
        use_mask=True,
        num_classes=7,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
            # 7-class weights: unknown boosted to penalize false free predictions
            # class 0(free):90.24% 1(unknown):2.91% 2(person):0.01%
            # 3(table):0.84% 4(chair):0.01% 5(floor):5.66% 6(car):0.32%
            class_weight=[1.0, 3.0, 2.173, 1.393, 2.247, 1.201, 1.515],
        ),
    )
)

# Data
dataset_type = 'FisheyeDatasetOccpancy'
data_root = '/home/qiaofeng/datasets/ros2_bag/fisheye_dataset/'

file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='FisheyePrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='FisheyeLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='FisheyeLoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        shift_height=False,
        file_client_args=file_client_args),
    dict(
        type='FisheyePointToMultiViewDepth',
        downsample=1,
        grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(
        type='FisheyePrepareImageInputs',
        data_config=data_config,
        sequential=False),
    dict(
        type='FisheyeLoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='FisheyeLoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        shift_height=False,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'fisheye_infos_val.pkl')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'fisheye_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24, ])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

load_from = None
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
