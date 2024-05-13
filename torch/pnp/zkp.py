import torch

def gt_zkp(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.tolist() > b.tolist()