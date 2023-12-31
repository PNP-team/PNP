.. role:: hidden
    :class: hidden-section

Tensor Parallelism - torch.distributed.tensor.parallel
======================================================

Tensor Parallelism(TP) is built on top of the PyTorch DistributedTensor
(`DTensor <https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md>`__)
and provides several parallelism styles: Rowwise and Colwise Parallelism.

.. warning ::
    Tensor Parallelism APIs are experimental and subject to change.

The entrypoint to parallelize your ``nn.Module`` using Tensor Parallelism is:

.. automodule:: torch.distributed.tensor.parallel

.. currentmodule:: torch.distributed.tensor.parallel

.. autofunction::  parallelize_module

Tensor Parallelism supports the following parallel styles:

.. autoclass:: torch.distributed.tensor.parallel.style.RowwiseParallel
  :members:

.. autoclass:: torch.distributed.tensor.parallel.style.ColwiseParallel
  :members:

.. warning::
    We are deprecating the styles below and will remove them soon:

.. autoclass:: torch.distributed.tensor.parallel.style.PairwiseParallel
  :members:

.. autoclass:: torch.distributed.tensor.parallel.style.SequenceParallel
  :members:

Since Tensor Parallelism is built on top of DTensor, we need to specify the
DTensor layout of the input and output of the module so it can interact with
the module parameters and module afterwards. Users can achieve this by specifying
the ``input_layouts`` and ``output_layouts`` which annotate inputs as DTensors
and redistribute the outputs, if needed.

If users only want to annotate the DTensor layout for inputs/outputs and no need to
distribute its parameters, the following classes can be used in the ``parallelize_plan``
of ``parallelize_module``:


.. currentmodule:: torch.distributed.tensor.parallel.style
.. autofunction::  PrepareModuleInput
.. autofunction::  PrepareModuleOutput

.. warning::
    We are deprecating the methods below and will remove them soon:
.. autofunction::  make_input_replicate_1d
.. autofunction::  make_input_reshard_replicate
.. autofunction::  make_input_shard_1d
.. autofunction::  make_input_shard_1d_last_dim
.. autofunction::  make_output_replicate_1d
.. autofunction::  make_output_reshard_tensor
.. autofunction::  make_output_shard_1d
.. autofunction::  make_output_tensor


Currently, there are some constraints which makes it hard for the ``MultiheadAttention``
module to work out of box for Tensor Parallelism, so we recommend users to try ``ColwiseParallel``
and ``RowwiseParallel`` for each parameter. There might be some code changes needed now
since we are parallelizing on the head dim of the ``MultiheadAttention`` module.


We also support 2D parallelism, where we compose tensor parallelism with data parallelism.
To integrate with ``FullyShardedDataParallel``,
users just need to call the following API explicitly:


.. currentmodule:: torch.distributed.tensor.parallel.fsdp
.. autofunction::  enable_2d_with_fsdp


To integrate with ``DistributedDataParallel``,
users just need to call the following API explicitly:


.. currentmodule:: torch.distributed.tensor.parallel.ddp
.. autofunction::  pre_dp_module_transform
