# configs/catseg/catseg_coco_stuff_single_file.py

# 1. 모델 설정
model = dict(
    type='CATSegWrapper',
    d2_yaml_cfg='configs/vitb_384.yaml',
    d2_weights_path='pretrained/model_base.pth', # https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor = 32
    )
)

# 2. 데이터셋 설정
dataset_type = 'COCOStuffDataset'
data_root = '/data/datasets/coco_stuff164k/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(1.0, 1.3), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs',
                 # 아래 meta_keys를 추가하여 필요한 모든 메타 정보를 명시합니다.
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_id'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train2017', seg_map_path='annotations/train2017'),
        pipeline=train_pipeline,
        seg_map_suffix=".png" # mmseg/datasets/coco_stuff.py __init__.py에서 seg_map_suffix='_labelTrainIds.png' 로 초기화. 변경
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# 3. 스케줄 및 런타임 설정
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float32',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2.5e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)
# optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=dict(max_norm=0.01, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=0.9, by_epoch=False, begin=1500, end=80000)
]
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict( project='CAT-SEG-REPRODUCE',
                           name='CAT-SEG-TRAIN',
                           entity='jasoncswoo-korea-university'),
         save_dir='wandb_logs')
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)
