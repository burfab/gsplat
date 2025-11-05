import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position, relocate, sample_add
from strategy import MCMCStrategy


@dataclass
class MyMultiSplatStrategy(Strategy):
    """Strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_

    This strategy will:

    - Periodically teleport GSs with low opacity to a place that has high opacity.
    - Periodically introduce new GSs sampled based on the opacity distribution.
    - Periodically perturb the GSs locations.

    Args:
        cap_max (int): Maximum number of GSs. Default to 1_000_000.
        noise_lr (float): MCMC samping noise learning rate. Default to 5e5.
        refine_start_iter (int): Start refining GSs after this iteration. Default to 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default to 25_000.
        refine_every (int): Refine GSs every this steps. Default to 100.
        min_opacity (float): GSs with opacity below this value will be pruned. Default to 0.005.
        verbose (bool): Whether to print verbose information. Default to False.

    Examples:

        >>> from gsplat import MCMCStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = MCMCStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info, lr=1e-3)

    """

    strategies : Dict[str, Strategy] = {
        "free": MCMCStrategy(),
        "mesh": Strategy()
    }

    def initialize_state(self, splats: Dict[str, int]) -> Dict[str,Dict[str, Any]]:
        """Initialize and return the running state for this strategy."""
        state = dict()
        for s in splats:
            state[s] = self.strategies[s].initialize_state()
        return state
    
    def required_param_keys(self):
        return ["means", "scales", "quats", "opacities"]

    def check_sanity(
        self,
        splats: Dict[str,int],
        params: Dict[str,Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]],
        optimizers: Dict[str,Dict[str, torch.optim.Optimizer]],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """
        for s in self.strategies:
            assert s in splats, "Needs " + s + " splat"
            super().check_sanity(params[s], optimizers[s])
            # The following keys are required for this strategy.
            for key in self.required_param_keys():
                assert key in params[s], f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        splats: Dict[str,int],
        params: Dict[str,Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]],
        optimizers: Dict[str,Dict[str, torch.optim.Optimizer]],
        state: Dict[str,Dict[str, Any]],
        step: int,
        info: Dict[str, Any],
    ):
        for s in splats:
            self.strategies[s].step_pre_backward(params=params[s], optimizers=optimizers[s], state=state[s], step=step, info=info)
            

    def step_post_backward(
        self,
        splats: Dict[str,int],
        params: Dict[str, Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]],
        optimizers: Dict[str,Dict[str, torch.optim.Optimizer]],
        state: Dict[str,Dict[str, Any]],
        step: int,
        info: Dict[str, Any],
        lr: float,
    ):
        """Callback function to be executed after the `loss.backward()` call.

        Args:
            lr (float): Learning rate for "means" attribute of the GS.
        """
        for s in splats:
            self.strategies[s].step_post_backward(params=params[s], optimizers=optimizers[s], state=state[s], step=step, info=info, lr=lr)
            

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.min_opacity
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs
