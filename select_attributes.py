import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from cat_seg.config import add_cat_seg_config # CAT-Seg의 config를 로드하기 위함

def setup(config_file):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    add_cat_seg_config(cfg) # CAT-Seg 커스텀 config 추가
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def main(config_file, input_model_path, output_model_path):
    """
    학습된 모델을 로드하고, select_att()를 실행한 후,
    Attribute가 선택된 버전의 모델을 새로운 경로에 저장합니다.
    """
    cfg = setup(config_file)
    model = build_model(cfg)
    
    # 학습된 모델(.pth) 로드
    DetectionCheckpointer(model).load(input_model_path)
    print(f"Loaded trained model from {input_model_path}")
    
    # Attribute 선택 메소드 실행
    # per_class는 데이터셋에 맞게 조절 가능
    model.select_att(per_class=25)
    
    # 선택된 Attribute가 적용된 모델 state_dict 저장
    torch.save(model.state_dict(), output_model_path)
    print(f"Saved model with selected attributes to {output_model_path}")


if __name__ == "__main__":
    # --- 설정 ---
    CONFIG_FILE = "configs/coco-stuff164k_vitb_384_ow_ovss.yaml"
    # 1단계에서 학습이 완료된 모델 경로
    INPUT_MODEL_PATH = "output/model_final.pth" 
    # Attribute 선택 후 새로 저장될 모델 경로
    OUTPUT_MODEL_PATH = "output/model_final_with_selected_atts.pth"
    # --- 실행 ---
    main(CONFIG_FILE, INPUT_MODEL_PATH, OUTPUT_MODEL_PATH)