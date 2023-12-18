import math
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .module import Module
from .lazy import LazyModuleMixin


__all__ = [
    'NTT_zkp'
]


class NTT_zkp(Module):
    r"""Applies a Number Theory Transformation(NTT) for a 2-dim tensor`

    This module only supports all curve types, does not support common types.

    Args:
        Params: a tensor that stores all parameters required in the NTT process.

    Attributes:
        is_intt: NTT direction, forward(False) or inverse(True).
        gpu_id: gpu device number used.
        domain_size: root of unity size in Fr of the selected curve.
        dtype: specified elliptic curve.

    Examples::

        >>> m = nn.NTT_zkp(True, 0, domain_size, torch.BLS12_377_Fr_G1_Mont)
        >>> random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
        >>> x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Mont)
        >>> input = x.to("cuda")
        >>> output = m.forward(input, True, False)

    """
    __constants__ = ['Params']

    Params: Tensor
    
    def __init__(self, is_intt: bool, gpu_id: int, domain_size: int, dtype) -> None:
        super().__init__()
        self.Params = torch.params_zkp(domain_size, gpu_id, is_intt, dtype = dtype, device='cuda')


    def forward(self, input: Tensor, is_intt: bool, is_coset: bool) -> Tensor:
        if is_intt & is_coset:
            output = torch.intt_coset_zkp(input, self.Params)
        elif is_intt:
            output = torch.intt_zkp(input, self.Params)
        elif is_coset:
            output = torch.ntt_coset_zkp(input, self.Params)
        else:
            output = torch.ntt_zkp(input, self.Params)
        return output


    
