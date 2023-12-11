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
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['Params']
    
    # windows_size: int
    # gpu_id: int
    # is_intt: bool
    Params: Tensor
    


    def __init__(self, is_intt: bool, gpu_id: int, domain_size: int) -> None:
        super().__init__()
        self.Params = torch.params_zkp(domain_size, gpu_id, is_intt, dtype= torch.BLS12_381_Fr_G1_Mont, device='cuda')


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


    
