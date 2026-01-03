import math
from typing import Tuple

import torch
from torch import Tensor

from .cuda._wrapper import _make_lazy_cuda_func


def compute_relocation(
    opacities: Tensor,  # [N]
    scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
    binoms: Tensor,  # [n_max, n_max]
) -> Tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        opacities: The opacities of the Gaussians. [N]
        scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]
        binoms: Precomputed lookup table for binomial coefficients used in
          Equation 9 in the paper. [n_max, n_max]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """

    N = opacities.shape[0]
    n_max, _ = binoms.shape
    isotropic = scales.shape[1] == 1
    assert scales.shape == (N, 3) or scales.shape == (N,1), scales.shape
    assert ratios.shape == (N,), ratios.shape
    opacities = opacities.contiguous()
    scales = scales
    ratios.clamp_(min=1, max=n_max)
    ratios = ratios.int().contiguous()

    new_opacities, new_scales = _make_lazy_cuda_func("relocation")(
        opacities, scales.expand((-1,3)).contiguous(), ratios, binoms, n_max
    )
    if isotropic: new_scales = new_scales.max(dim=1,keepdim=True).values
    return new_opacities, new_scales
