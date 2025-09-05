# configs/catseg/catseg_coco_stuff_single_file.py

# 1. Model Setting
model = dict(
    type='CATSegWrapper',
    d2_yaml_cfg='configs/vitb_384.yaml',
    #d2_weights_path='pretrained/model_base.pth', 
    #https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth
    d2_weights_path=None,
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.280, 103.530],
        std=[58.395, 57.120, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size_divisor = 32))


# 2. Datasets Setting
# <data_root>
dataset_type = 'COCOStuffDataset'
data_root = '/data/datasets/coco_stuff164k/'

# <Pipeline>
train_pipeline = [dict(type='LoadImageFromFile'),
                  dict(type='LoadAnnotations'),
                  # dict(type='RandomResize', scale=(512, 512), ratio_range=(1.0, 1.3), keep_ratio=True),
                  # dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
                  dict(type='RandomResize', scale=(384, 384), ratio_range=(1.0, 1.3), keep_ratio=True),
                  dict(type='RandomCrop', crop_size=(384, 384), cat_max_ratio=0.75, ignore_index=255,),        
                  dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                  #dict(type='PhotoMetricDistortion'),
                  dict(type='PackSegInputs',
                  # 아래 meta_keys를 추가하여 필요한 모든 메타 정보를 명시합니다.
                  meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                             'pad_shape', 'scale_factor', 'flip', 'flip_direction','img_id'))]

test_pipeline = [dict(type='LoadImageFromFile'),
                 dict(type='LoadAnnotations'),
                 dict(type='Resize', scale=(384, 384), keep_ratio=True),
                 dict(type='PackSegInputs',
                      meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                                 'pad_shape', 'scale_factor', 'flip', 'flip_direction','img_id'))]

# <DataLoader>
train_dataloader = dict(batch_size=2,
                        num_workers=4,
                        persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                    #  indices=list(range(100)),
                                     data_prefix=dict(img_path='images/train2017', seg_map_path='annotations_detectron2/train2017'),
                                     pipeline=train_pipeline,
                                     # mmseg/datasets/coco_stuff.py __init__.py에서 seg_map_suffix='_labelTrainIds.png' 로 초기화. 변경
                                     seg_map_suffix=".png"))
val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                #    indices=list(range(100)),
                                   data_prefix=dict(img_path='images/val2017', seg_map_path='annotations_detectron2/val2017'),
                                   pipeline=test_pipeline,
                                   seg_map_suffix=".png"))

test_dataloader = val_dataloader

# <Evaluator>
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator


# 3. Scheduler & Runtime

# <Scheduler>
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(type='AmpOptimWrapper',
                     dtype='bfloat16',
                     loss_scale='dynamic',
                     optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
                     paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.0),  # 백본 가중치 동결
                                                     'text_encoder': dict(lr_mult=0.01) # CLIP 텍스트 인코더는 0.01배의 학습률 적용
                                                     }),
                     clip_grad=dict(max_norm=1.0, norm_type=2))
# optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=dict(max_norm=0.01, norm_type=2))

# param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#                    dict(type='PolyLR', power=0.9, by_epoch=False, begin=1500, end=80000)]

param_scheduler = [
    # 웜업(Warmup) 스케줄러: 1,500번의 반복 동안 학습률을 선형적으로 증가시킵니다.
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    # 메인 스케줄러: CosineAnnealingLR
    dict(type='CosineAnnealingLR',
         T_max=80000,  # 총 반복 횟수
         eta_min=0.0,    # 최소 학습률
         by_epoch=False,
         begin=1500,
         end=80000)]

default_scope = 'mmseg'
env_cfg = dict(cudnn_benchmark=True,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))


# <Runtimes>
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend',
                     init_kwargs=dict(project='CAT-SEG-REPRODUCE',
                                      name='CAT-SEG-TRAIN',
                                      entity='jasoncswoo-korea-university'),
                     save_dir='wandb_logs')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
default_hooks = dict(timer=dict(type='IterTimerHook'),
                     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
                     param_scheduler=dict(type='ParamSchedulerHook'),
                     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
                     sampler_seed=dict(type='DistSamplerSeedHook'),
                     visualization=dict(type='SegVisualizationHook'))