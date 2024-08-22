from .generator import Yolov8Generator
from .model_onnx import Yolov8SegmentationONNX
# from .model_torchscript import Yolov8SegmentationTorchscript

from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any


class Yolov8Config(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())

    model_path: str
    server: str = 'yolov8'
    device: str = 'cpu'
    batch_size: int = 4
    default_input_size: int = 640
    dzi_settings: Dict[str, Any] = {
        'format': 'jpeg', 
        'tile_size': 512, 
        'overlap': 64, 
        'limit_bounds': False, 
        'tile_quality': 50,
    }
    labels: List[str] = [
        'bg', 'tumor_nuclei', 'stromal_nuclei', 'immune_nuclei', 
        'blood_cell', 'macrophage', 'dead_nuclei', 'other_nuclei',
    ]
    labels_color: Dict[str, str] = {
        -100: "#949494", 
        0: "#ffffff", 
        1: "#00ff00", 
        2: "#ff0000", 
        3: "#0000ff", 
        4: "#ff00ff", 
        5: "#ffff00",
        6: "#0094e1",
        7: "#646464",
    }
    labels_text: Dict[int, str] = {
        0: 'bg', 1: 'tumor_nuclei', 2: 'stromal_nuclei', 3: 'immune_nuclei', 
        4: 'blood_cell', 5: 'macrophage', 6: 'dead_nuclei', 7: 'other_nuclei',
    }
    nms_params: Dict[str, Any] = {
        "conf_thres": 0.15, #  0.25,  # score_threshold, discards boxes with score < score_threshold
        "iou_thres": 0.45, #  0.7,  # iou_threshold, discards all overlapping boxes with IoU > iou_threshold
        "classes": None, 
        "agnostic": True, # False
        "multi_label": False, 
        "labels": (), 
        "nc": 7,
        "max_det": 300,  # maximum detection
    }

__all__ = [
    'Yolov8Generator', 
    'Yolov8SegmentationONNX',
    'Yolov8Config', 
]
