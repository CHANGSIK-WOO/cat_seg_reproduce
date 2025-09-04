# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from cat_seg.cat_seg_model import CATSeg
from cat_seg.config import add_cat_seg_config

from mmseg.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.structures import SegDataSample
from mmseg.utils import (ForwardResults, 
                         OptConfigType, OptMultiConfig, OptSampleList, SampleList)
from mmengine.structures import InstanceData, PixelData


from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer


# mmsegmentation's (img, SegDataSample) -> D2's
def _pack_from_mmseg(inputs, data_samples, device):
    batched = []
    for img, sample in zip(inputs, data_samples):
        img_h, img_w = img.shape[-2:]   # fallback
        ori_h, ori_w = img_h, img_w

        if hasattr(sample, 'metainfo'):
            meta = sample.metainfo
            if isinstance(meta, dict):
                if 'img_shape' in meta:
                    img_h, img_w = meta['img_shape'][:2]
                if 'ori_shape' in meta:
                    ori_h, ori_w = meta['ori_shape'][:2]
        elif isinstance(sample, dict):
            meta = sample.get('metainfo', sample)
            if isinstance(meta, dict):
                if 'img_shape' in meta:
                    img_h, img_w = meta['img_shape'][:2]
                if 'ori_shape' in meta:
                    ori_h, ori_w = meta['ori_shape'][:2]

        if hasattr(sample, 'gt_sem_seg'):
            # D2의 MaskFormerSemanticDatasetMapper는 gt를 'sem_seg' 키에 저장합니다.
            sem_seg = sample.gt_sem_seg.data.to(device)
        else:
            sem_seg = None
        
        batched.append({
            'image': img.to(device),
            'sem_seg': sem_seg,
            'height': img_h, 'width': img_w,
            'ori_height': ori_h, 'ori_width': ori_w,
        })

    return batched

# D2's Predict Result -> mmsegmentation's (img, SegDataSample)
def _to_mmseg(processed_results, data_samples, device):
    """D2 processed_results(list[{'instances': Instances}]) -> MMDet DetDataSample"""
    outs = []
    for pred, sample in zip(processed_results, data_samples):
        pred_sem_seg_logits = pred['sem_seg'] # (Num_classes, H, W)
        ds = SegDataSample()
        ds.pred_sem_seg = PixelData(data=pred_sem_seg_logits)

        if hasattr(sample, 'metainfo'):
            ds.set_metainfo(sample.metainfo)

        outs.append(ds)
    return outs

@MODELS.register_module()
class CATSegWrapper(BaseSegmentor):
    def __init__(self, d2_yaml_cfg, d2_weights_path=None, init_cfg=None, data_preprocessor=None):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        cfg = get_cfg()
        add_cat_seg_config(cfg)
        cfg.merge_from_file(d2_yaml_cfg)
        cfg.freeze()

        self.d2_model = CATSeg(cfg)
        self.device = self.d2_model.device
        if d2_weights_path:
            # d2_model을 위한 checkpointer를 생성합니다.
            checkpointer = DetectionCheckpointer(self.d2_model)
            # 가중치를 로드합니다.
            checkpointer.load(d2_weights_path)
            print(f"Loaded Detectron2 weights from: {d2_weights_path}")        

    def extract_feat(self, inputs): 
        """d2_model이 backbone 역할을 겸하므로 이 메서드는 사용되지 않습니다."""
        raise NotImplementedError('CATSegWrapper does not support separate feature extraction.')        

    def encode_decode(self, inputs, batch_data_samples):
        """d2_model이 전체 encode-decode를 수행하므로 이 메서드는 사용되지 않습니다."""
        raise NotImplementedError('CATSegWrapper does not support separate encode_decode.')

    def forward(self, inputs: Tensor, data_samples, mode='tensor'):
        if mode == 'loss': return self.loss(inputs, data_samples)
        elif mode == 'predict': return self.predict(inputs, data_samples)
        elif mode == 'tensor': return self._forward(inputs, data_samples)
        else: raise RuntimeError(f'Invalid mode "{mode}". ''Only supports loss, predict and tensor mode')

    def loss(self, inputs, data_samples):
        batched_inputs = _pack_from_mmseg(inputs, data_samples, self.device) # mmseg -> d2
        self.d2_model.train() #train mode 
        losses = self.d2_model(batched_inputs)
        
        return losses
    
    def predict(self, inputs, data_samples):
        batched_inputs = _pack_from_mmseg(inputs, data_samples, self.device) # mmseg -> d2
        self.d2_model.eval() #eval mode
        with torch.no_grad():
            processed_results = self.d2_model(batched_inputs)
        
        results = _to_mmseg(processed_results, data_samples, self.device)

        return results
    
    def _forward(self, inputs, data_samples):
        batched_inputs = _pack_from_mmseg(inputs, data_samples, self.device) # mmseg -> d2
        self.d2_model.eval()
        with torch.no_grad():
            outputs = self.d2_model(batched_inputs)
        
        # sem_seg 텐서 자체(logit)를 반환
        return [res['sem_seg'] for res in outputs]        
               
