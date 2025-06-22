point_cloud_range = [-76.8, -76.8, -75.0, 76.8, 76.8, -67.0]
class_names = ['car', 'pedestrian']
dataset_type = 'NuScenesDataset'
data_root = '/mnt/data/wzc/det/data/nuscenes'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=dict(
            cams=['UAV_1_1', 'UAV_1_2', 'UAV_1_3', 'UAV_2', 'UAV_3'],
            Ncams=5,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0,
            noise_std=0)),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=['car', 'pedestrian']),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-76.8, -76.8, -75.0, 76.8, 76.8, -67.0]),
    dict(type='ObjectNameFilter', classes=['car', 'pedestrian']),
    dict(type='DefaultFormatBundle3D', class_names=['car', 'pedestrian']),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=dict(
            cams=['UAV_1_1', 'UAV_1_2', 'UAV_1_3', 'UAV_2', 'UAV_3'],
            Ncams=5,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0,
            noise_std=0)),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=['car', 'pedestrian'],
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=['car', 'pedestrian'],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            data_root='/mnt/data/wzc/det/data/nuscenes',
            ann_file='/mnt/data/wzc/det/data/nuscenesnuscenes_infos_train.pkl',
            pipeline=[
                dict(
                    type='PrepareImageInputs',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'UAV_1_1', 'UAV_1_2', 'UAV_1_3', 'UAV_2', 'UAV_3'
                        ],
                        Ncams=5,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.0,
                        noise_std=0)),
                dict(
                    type='LoadAnnotationsBEVDepth',
                    bda_aug_conf=dict(
                        rot_lim=(-22.5, 22.5),
                        scale_lim=(0.95, 1.05),
                        flip_dx_ratio=0.5,
                        flip_dy_ratio=0.5),
                    classes=['car', 'pedestrian']),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-76.8, -76.8, -75.0, 76.8, 76.8,
                                       -67.0]),
                dict(type='ObjectNameFilter', classes=['car', 'pedestrian']),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=['car', 'pedestrian']),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            classes=['car', 'pedestrian'],
            test_mode=False,
            use_valid_flag=False,
            box_type_3d='LiDAR',
            type='NuScenesDataset',
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='/mnt/data/wzc/det/data/nuscenesnuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=['UAV_1_1', 'UAV_1_2', 'UAV_1_3', 'UAV_2', 'UAV_3'],
                    Ncams=5,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0,
                    noise_std=0)),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=['car', 'pedestrian'],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['car', 'pedestrian'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img_inputs'])
                ])
        ],
        classes=['car', 'pedestrian'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='/mnt/data/wzc/det/data/nuscenesnuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=['UAV_1_1', 'UAV_1_2', 'UAV_1_3', 'UAV_2', 'UAV_3'],
                    Ncams=5,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0,
                    noise_std=0)),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=['car', 'pedestrian'],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['car', 'pedestrian'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img_inputs'])
                ])
        ],
        classes=['car', 'pedestrian'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './debug/bevdet-r50-cbgs'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
data_config = dict(
    cams=['UAV_1_1', 'UAV_1_2', 'UAV_1_3', 'UAV_2', 'UAV_3'],
    Ncams=5,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0,
    noise_std=0)
grid_config = dict(
    x=[-76.8, 76.8, 0.8],
    y=[-76.8, 76.8, 0.8],
    z=[-75.0, -67.0, 8.0],
    depth=[20.0, 160.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80
multi_adj_frame_id_cfg = (1, 2, 1)
model = dict(
    type='BEVDet',
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=dict(
            x=[-76.8, 76.8, 0.8],
            y=[-76.8, 76.8, 0.8],
            z=[-75.0, -67.0, 8.0],
            depth=[20.0, 160.0, 1.0]),
        input_size=(256, 704),
        in_channels=256,
        out_channels=80,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet', numC_input=80, num_channels=[160, 320, 640]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=800, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[dict(num_class=2, class_names=['car', 'pedestrian'])],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[-76.8, -76.8],
            post_center_range=[-61.2, -61.2, -75.0, 61.2, 61.2, -67.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-76.8, -76.8, -75.0, 76.8, 76.8, -67.0],
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=[-76.8, -76.8],
            post_center_limit_range=[-86.8, -86.8, -75.0, 86.8, 86.8, -67.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=500,
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[
                1.0, 0.7, 0.7, 0.4, 0.55, 1.1, 1.0, 1.0, 1.5, 3.5
            ]])))
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
share_data_config = dict(
    type='NuScenesDataset',
    classes=['car', 'pedestrian'],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet')
test_data_config = dict(
    pipeline=[
        dict(
            type='PrepareImageInputs',
            data_config=dict(
                cams=['UAV_1_1', 'UAV_1_2', 'UAV_1_3', 'UAV_2', 'UAV_3'],
                Ncams=5,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0,
                noise_std=0)),
        dict(
            type='LoadAnnotationsBEVDepth',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=['car', 'pedestrian'],
            is_train=False),
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=['car', 'pedestrian'],
                    with_label=False),
                dict(type='Collect3D', keys=['points', 'img_inputs'])
            ])
    ],
    ann_file='/mnt/data/wzc/det/data/nuscenesnuscenes_infos_val.pkl',
    type='NuScenesDataset',
    classes=['car', 'pedestrian'],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet')
key = 'test'
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=20)
custom_hooks = [
    dict(type='MEGVIIEMAHook', init_updates=10560, priority='NORMAL')
]
gpu_ids = range(0, 2)
