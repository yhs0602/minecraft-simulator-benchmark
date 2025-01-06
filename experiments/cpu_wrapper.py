from typing import SupportsFloat, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType


# This wrapper converts the gpu tensor from craftground zero-copy mode into a cpu tensor, for compatibility with the
# gym interface. This is necessary because the gym interface expects numpy arrays, which are not supported by the
# zero-copy mode.
# This wrapper expects the vision only observation.
class CPUVisionWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        self.env = env
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs.cpu(),
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.cpu(), info
