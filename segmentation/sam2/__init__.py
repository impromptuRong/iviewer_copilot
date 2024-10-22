from .sam2 import SAM2Segmentation
from typing import Tuple

from pydantic import BaseModel, ConfigDict


class SAM2Config(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())

    model_path: str
    server: str = 'sam2'
    device: str = 'cpu'
    default_input_size: Tuple = (512, 512)


__all__ = [
    'SAM2Segmentation',
    'SAM2Config', 
]
