# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple


import math
import os
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange

@META_ARCH_REGISTRY.register()
class OWOVSSCATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,

        #=====ow-ovd=====#
        att_embeddings_path: str = None,
        distributions_path: str = None,
        thr: float = 0.8,
        alpha: float = 0.5,
        top_k: int = 10,

    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))

        # --- OW-OVD 로직 추가 ---
        self.thr = thr
        self.alpha = alpha
        self.top_k = top_k
        self.distributions_path = distributions_path
        self.device = self.pixel_mean.device
        self.thrs = [thr] # 분포 저장을 위한 임계값 리스트
        
        self.load_att_embeddings(att_embeddings_path)
        self.enable_log() # 분포 수집 활성화
        # --- OW-OVD 로직 추가 끝 ---            


    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            # --- OW-OVD 로직 추가 ---
            "att_embeddings_path": cfg.MODEL.SEM_SEG_HEAD.ATT_EMBEDDINGS_PATH,
            "distributions_path": cfg.MODEL.SEM_SEG_HEAD.DISTRIBUTIONS_PATH,
            "thr": cfg.MODEL.SEM_SEG_HEAD.UNKNOWN_THR,
            "alpha": cfg.MODEL.SEM_SEG_HEAD.ALPHA,
            "top_k": cfg.MODEL.SEM_SEG_HEAD.TOP_K,
            # --- OW-OVD 로직 추가 끝 ---            
        }
    # --- OW-OVD 로직 추가 ---
    # OW-OVD의 our_head.py에서 Attribute 관련 핵심 함수들을 가져와 현 클래스에 맞게 수정합니다.

    def disable_log(self):
        self.positive_distributions = None
        self.negative_distributions = None
        print('disable log : distributions to None')
    
    def enable_log(self):
        self.reset_log()
        print('enable log : distribution made')

    def load_att_embeddings(self, att_embeddings_path):
        if att_embeddings_path is None or not os.path.exists(att_embeddings_path):
            self.att_embeddings = None
            self.all_atts = None
            self.texts = None
            self.disable_log()
            print("Attribute embeddings not found. Disabling unknown detection.")
            return

        atts = torch.load(att_embeddings_path)
        self.texts = atts['att_text']
        self.all_atts = atts['att_embedding']
        self.att_embeddings = self.all_atts.clone().float()

    def reset_log(self, interval=0.0001):
        if self.att_embeddings is None:
            return
        print('Resetting attribute distributions log.')
        self.positive_distributions = [{att_i: torch.zeros(int(1 / interval)).to(self.device)
                                    for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        self.negative_distributions = [{att_i: torch.zeros(int(1 / interval)).to(self.device)
                                      for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]

    def get_sim(self, a, b):
        def jensen_shannon_divergence(p, q):
            m = 0.5 * (p + q)
            m = m.clamp(min=1e-6)
            js_div = 0.5 * (torch.sum(p * torch.log((p / m).clamp(min=1e-6))) +
                            torch.sum(q * torch.log((q / m).clamp(min=1e-6))))
            return js_div
        return jensen_shannon_divergence(a, b)

    def get_all_dis_sim(self, positive_dis, negative_dis):
        dis_sim = []
        # len(positive_dis) 대신 .keys()를 사용하여 순회하는 것이 더 안전합니다.
        for i in range(len(positive_dis)):
            positive = positive_dis[i]
            negative = negative_dis[i]
            positive = positive / (positive.sum() + 1e-9)
            negative = negative / (negative.sum() + 1e-9)
            dis_sim.append(self.get_sim(positive, negative))
        return torch.stack(dis_sim).to(self.device)

    def select_att(self, per_class=25):
        if self.att_embeddings is None or self.distributions_path is None:
            return
        
        print("Selecting attributes...")
        distributions = torch.load(self.distributions_path, map_location=self.device)
        self.positive_distributions, self.negative_distributions = distributions['positive_distributions'], distributions['negative_distributions']
        
        thr_id = self.thrs.index(self.thr)
        distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id], self.negative_distributions[thr_id])
        
        all_atts = self.all_atts.to(self.device)
        att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)
        cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).sigmoid()
        
        num_classes = len(self.sem_seg_head.predictor.test_class_names)
        total_to_select = min(per_class * num_classes, len(self.texts))
        selected_indices = []
        
        for _ in range(total_to_select):
            if len(selected_indices) == 0:
                idx = int(torch.argmin(distribution_sim).item())
            else:
                unselected_indices = list(set(range(len(self.texts))) - set(selected_indices))
                cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(dim=1)
                distribution_sim_unselected = distribution_sim[unselected_indices]
                score = self.alpha * distribution_sim_unselected + (1 - self.alpha) * cosine_sim_with_selected
                idx = unselected_indices[score.argmin()]
            
            selected_indices.append(idx)
        
        selected_indices = torch.tensor(selected_indices).to(self.device)
        self.att_embeddings = nn.Parameter(all_atts[selected_indices]).to(self.device)
        self.texts = [self.texts[i] for i in selected_indices]
        print(f"Selected {len(self.texts)} attributes.")

    def log_distribution(self, att_scores, targets):
        if not self.training or self.positive_distributions is None or self.att_embeddings is None:
            return

        # att_scores: (B, num_att, H, W), targets: (B, H, W)
        num_att = att_scores.shape[1]
        att_scores = att_scores.sigmoid().permute(0, 2, 3, 1).reshape(-1, num_att).float()
        
        # known_mask: 0, 1, ..., num_classes-1. unknown_mask: ignore_value
        # 여기서는 gt가 있는 픽셀을 positive, 없는 픽셀을 negative로 간주합니다.
        # OW-OVD의 'assigned_scores'와 유사한 역할을 하도록 gt_mask를 생성
        gt_mask = (targets != self.sem_seg_head.ignore_value).reshape(-1)

        for idx, thr in enumerate(self.thrs):
            # OW-OVD에서는 score가 thr 이상인 것을 positive로 보지만,
            # 여기서는 gt 존재 여부로 positive/negative를 나눕니다.
            positive_scores = att_scores[gt_mask]
            negative_scores = att_scores[~gt_mask]

            if positive_scores.numel() > 0:
                for att_i in range(num_att):
                    self.positive_distributions[idx][att_i] += torch.histc(positive_scores[:, att_i], bins=int(1/0.0001), min=0, max=1)
            if negative_scores.numel() > 0:
                for att_i in range(num_att):
                    self.negative_distributions[idx][att_i] += torch.histc(negative_scores[:, att_i], bins=int(1/0.0001), min=0, max=1)

    def calculate_uncertainty(self, known_logits):
        known_probs = torch.clamp(known_logits.sigmoid(), 1e-6, 1 - 1e-6)
        entropy = (-known_probs * torch.log(known_probs) - (1 - known_probs) * torch.log(1 - known_probs)).mean(dim=-1, keepdim=True)
        return entropy

    def compute_weighted_top_k_attributes(self, adjusted_scores: torch.Tensor, k: int = 10) -> torch.Tensor:
        top_k_scores, _ = adjusted_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_average = torch.sum(top_k_scores * top_k_weights, dim=-1, keepdim=True)
        return weighted_average

    def predict_unknown(self, known_outputs, clip_features, features):
        # known_outputs: (B, num_known_classes, H, W)
        
        # 1. Attribute score 계산
        # Attribute 임베딩을 predictor에 전달하여 attribute score를 얻습니다.
        # 이 부분은 predictor가 외부 임베딩을 받을 수 있도록 수정이 필요할 수 있습니다.
        # 여기서는 임시로 sem_seg_head를 다시 호출하는 것으로 시뮬레이션합니다.
        # 실제 구현 시에는 predictor의 forward를 수정하여 text embedding을 인자로 받는 것이 효율적입니다.
        att_text_features = self.sem_seg_head.predictor.get_text_features(self.texts, self.att_embeddings)
        unknown_outputs = self.sem_seg_head(clip_features, features, prompt=att_text_features)
        
        # 2. 로짓 형태 변환 및 확률 계산
        known_logits = known_outputs.permute(0, 2, 3, 1) # (B, H, W, num_known)
        unknown_logits = unknown_outputs.permute(0, 2, 3, 1) # (B, H, W, num_att)
        
        # 3. 불확실성(Uncertainty) 계산
        uncertainty = self.calculate_uncertainty(known_logits)

        # 4. Top-k Attribute Score 계산
        top_k_att_score = self.compute_weighted_top_k_attributes(unknown_logits.sigmoid(), k=self.top_k)

        # 5. Hybrid Fusion: Known score, Uncertainty, Attribute score 결합
        known_probs_max, _ = known_logits.sigmoid().max(-1, keepdim=True)
        unknown_score_final = (top_k_att_score + uncertainty) / 2 * (1 - known_probs_max)
        
        # 6. Known/Unknown 로짓 결합
        # 마지막 채널에 unknown score를 추가합니다.
        final_logits = torch.cat([known_logits, unknown_score_final], dim=-1)
        
        return final_logits.permute(0, 3, 1, 2) # (B, num_known + 1, H, W)
    # --- OW-OVD 로직 추가 끝 ---

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        self.layers = []

        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)

        image_features = clip_features[:, 1:, :]

        # CLIP ViT features for guidance
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3,}

        outputs = self.sem_seg_head(clip_features, features)
        orig_dtype = outputs.dtype
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0).squeeze(1) #torch.Size([2, 1, 512, 512]) -> [2, 512, 512]
            # print(targets.unique())
            #outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False) #torch.Size([2, 171, 512, 512])          
            # --- OW-OVD 로직 추가: 분포 수집 ---
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)

            if self.att_embeddings is not None:
                with torch.no_grad():
                    att_text_features = self.sem_seg_head.predictor.get_text_features(self.texts, self.att_embeddings)
                    att_scores = self.sem_seg_head(clip_features, features, prompt=att_text_features)
                    self.log_distribution(att_scores, targets)                  
            # --- OW-OVD 로직 추가 끝 ---
                        
            outputs = F.interpolate(outputs.float(), size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False).to(orig_dtype) #torch.Size([2, 171, 512, 512])          
            
            num_classes = outputs.shape[1]
            mask = (targets != self.sem_seg_head.ignore_value) #torch.Size([2, 512, 512])
            # print(mask.unique())            
            # print(targets[mask].unique())

            outputs = outputs.permute(0,2,3,1) # [2, 512, 512, 171]
            _targets = torch.zeros(outputs.shape, device=self.device) # [2, 512, 512, 171]
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float() 
            
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses

        else:
            # --- OW-OVD 로직 추가: Unknown 예측 ---
            if self.att_embeddings is not None:
                # known class 예측 결과와 피쳐를 predict_unknown 함수에 전달
                outputs = self.predict_unknown(outputs, clip_features, features)
            # --- OW-OVD 로직 추가 끝 ---
                        
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        features = {'res5': res5, 'res4': res4, 'res3': res3,}
        outputs = self.sem_seg_head(clip_features, features)

        # sliding window에서도 unknown 예측 로직을 추가합니다.
        if self.att_embeddings is not None:
            outputs = self.predict_unknown(outputs, clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
