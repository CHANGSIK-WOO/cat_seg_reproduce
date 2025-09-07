import os
import torch
from detectron2.engine.hooks import HookBase
from detectron2.utils import comm

class DistributionSaveHook(HookBase):
    def __init__(self, distributions_path):
        self._distributions_path = distributions_path

    def after_train(self):
        # 마스터 프로세서에서만 실행
        if comm.is_main_process():
            # 모델에서 수집된 분포 가져오기
            positive_distributions = self.trainer.model.positive_distributions
            negative_distributions = self.trainer.model.negative_distributions

            if positive_distributions is not None and negative_distributions is not None:
                print(f"Saving collected distributions to {self._distributions_path}...")
                
                # 저장 경로의 디렉토리가 없으면 생성
                os.makedirs(os.path.dirname(self._distributions_path), exist_ok=True)
                
                # 분포 저장
                torch.save({
                    'positive_distributions': positive_distributions,
                    'negative_distributions': negative_distributions
                }, self._distributions_path)
                print("Distributions saved successfully.")