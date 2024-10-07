from typing import NamedTuple

import torch


class DatasetItem(NamedTuple):
    img: torch.Tensor
    pos: torch.Tensor = torch.zeros(0)
    size: torch.Tensor = torch.zeros(0)
    id: torch.Tensor = torch.zeros(0)
    in_camera: torch.Tensor = torch.zeros(0)
    action: torch.Tensor = torch.zeros(0)
