from typing import Any

import torch
from torch import Tensor
from .module import Module
import torch.nn.functional as F 

__all__ = ["Ntt", "Intt", "Ntt_coset", "Intt_coset"]


class Ntt(Module):
    r"""Applies a Number Theory Transformation(NTT) for a 2-dim tensor`

    This module only supports all curve types, does not support common types.

    Args:
        Params: a tensor that stores all parameters required in the NTT process.

    Attributes:
        is_intt: NTT direction, forward(False) or inverse(True).
        domain_size: root of unity size in Fr over the selected curve.
        dtype: specified elliptic curve.

    Examples::

        >>> m = nn.ntt(True, domain_size, torch.BLS12_377_Fr_G1_Mont)
        >>> random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
        >>> x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Mont)
        >>> input = x.to("cuda")
        >>> output = m.forward(input)

    """

    __constants__ = ["Params"]

    Params: Tensor

    def __init__(self, domain_size: int, dtype) -> None:
        super().__init__()
        self.Params = torch.params_zkp(
            domain_size, is_intt=False, dtype=dtype, device="cuda"
        )

    def forward(self, input: Tensor) -> Tensor:
        output = torch.ntt_zkp(input, self.Params, is_intt=False, is_coset=False)
        return output


class Intt(Module):
    r"""Applies a Number Theory Transformation(NTT) for a 2-dim tensor`

    This module only supports all curve types, does not support common types.

    Args:
        Params: a tensor that stores all parameters required in the NTT process.

    Attributes:
        domain_size: root of unity size in Fr over the selected curve.
        dtype: specified elliptic curve.

    Examples::

        >>> m = nn.intt(domain_size, torch.BLS12_377_Fr_G1_Mont)
        >>> random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
        >>> x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Mont)
        >>> input = x.to("cuda")
        >>> output = m.forward(input)

    """

    __constants__ = ["Params"]

    Params: Tensor

    def __init__(self, domain_size: int, dtype) -> None:
        super().__init__()
        self.Params = torch.params_zkp(
            domain_size, is_intt=True, dtype=dtype, device="cuda"
        )

    def forward(self, input: Tensor) -> Tensor:
        output = torch.ntt_zkp(input, self.Params, is_intt=True, is_coset=False)
        return output


class Ntt_coset(Module):
    r"""Applies a Number Theory Transformation(NTT) for a 2-dim tensor`

    This module only supports all curve types, does not support common types.

    Args:
        Params: a tensor that stores all parameters required in the NTT process.

    Attributes:
        is_intt: NTT direction, forward(False) or inverse(True).
        domain_size: root of unity size in Fr over the selected curve.
        dtype: specified elliptic curve.

    Examples::

        >>> m = nn.ntt_coset(domain_size, torch.BLS12_377_Fr_G1_Mont)
        >>> random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
        >>> x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Mont)
        >>> input = x.to("cuda")
        >>> output = m.forward(input)

    """

    __constants__ = ["Params","Size"]

    Params: Tensor
    Size: int

    def __init__(self, domain_size: int, coset_size: int, dtype) -> None:
        super().__init__()
        self.Params = torch.params_zkp(
            domain_size, is_intt=False, dtype = dtype, device="cuda"
        )
        self.Size = coset_size

    def forward(self, input: Tensor) -> Tensor:
        input = F.resize_poly(input, self.Size)
        output = torch.ntt_zkp(input, self.Params, is_intt=False, is_coset=True)
        return output


class Intt_coset(Module):
    r"""Applies a Number Theory Transformation(NTT) for a 2-dim tensor`

    This module only supports all curve types, does not support common types.

    Args:
        Params: a tensor that stores all parameters required in the NTT process.

    Attributes:
        domain_size: root of unity size in Fr over the selected curve.
        dtype: specified elliptic curve.

    Examples::

        >>> m = nn.intt_coset(domain_size, torch.BLS12_377_Fr_G1_Mont)
        >>> random_list = [[random.randint(1, 1000) for _ in range(4)] for _ in range(1024)]
        >>> x = torch.tensor(random_list, dtype=torch.BLS12_377_Fr_G1_Mont)
        >>> input = x.to("cuda")
        >>> output = m.forward(input)

    """

    __constants__ = ["Params"]

    Params: Tensor

    def __init__(self, domain_size: int, dtype) -> None:
        super().__init__()
        self.Params = torch.params_zkp(
            domain_size, is_intt=True, dtype=dtype, device="cuda"
        )

    def forward(self, input: Tensor) -> Tensor:
        output = torch.ntt_zkp(input, self.Params, is_intt=True, is_coset=True)
        return output

class MSM(Module):
    pass 
    # __constants__ = ["dtype"]

    # dtype: Any

    # def __init__(self, bases: Tensor, scalar: Tensor, dtype) -> None:
    #     super().__init__()
    #     self.dtype = dtype

    # def forward(self, bases: Tensor, scalar: Tensor) -> Tensor:
    #     output = torch.multi_scalar_mult(bases.to("cuda"), scalar.to("cuda"))
    #     return output
    

