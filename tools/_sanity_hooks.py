# file: tools/_sanity_hooks.py
import json
from mmengine.hooks import Hook
from mmengine.dist import is_main_process

class ClassMappingSanityHook(Hook):
    def before_train(self, runner):
        if is_main_process():
            self._check(runner, when="before_train")

    def before_val(self, runner):
        if is_main_process():
            self._check(runner, when="before_val")

    def _check(self, runner, when):
        # 1) MMSEG classes
        ds = runner.val_dataloader.dataset
        metas = getattr(ds, 'METAINFO', getattr(ds, 'metainfo', {}))
        mmseg_classes = list(metas.get('classes', []))

        # 2) model num_classes
        from mmengine.model import is_model_wrapper
        mdl = runner.model.module if is_model_wrapper(runner.model) else runner.model
        num_out = getattr(getattr(mdl, 'decode_head', None), 'num_classes', None) \
                  or getattr(mdl, 'num_classes', None)

        # 3) D2 classes
        d2_classes = json.load(open("datasets/coco.json"))

        print(f"[SANITY:{when}] mmseg={len(mmseg_classes)} model={num_out} d2={len(d2_classes)}")
        assert num_out == len(mmseg_classes) == len(d2_classes), "num_classes mismatch"

        # 순서까지 동일 확인(앞/뒤 샘플)
        assert mmseg_classes[:20] == d2_classes[:20] and mmseg_classes[-20:] == d2_classes[-20:], \
            "class order mismatch"
