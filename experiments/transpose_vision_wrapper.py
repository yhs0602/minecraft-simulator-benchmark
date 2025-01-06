from typing import SupportsFloat, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType
from torch import Tensor


# Transpose Vision wrapper (H, W, C) -> (C, H, W)
class TransposeVisionWrapper(gym.Wrapper):
    def __init__(self, env, x_dim, y_dim, **kwargs):
        self.env = env
        super().__init__(self.env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, y_dim, x_dim),
            dtype=np.uint8,
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = info["rgb"]
        rgb = self.transpose_rgb(rgb)
        return (
            rgb,
            reward,
            terminated,
            truncated,
            info,
        )

    def transpose_rgb(self, rgb):
        if isinstance(rgb, np.ndarray):
            rgb = np.transpose(rgb, (2, 0, 1))
        elif isinstance(rgb, Tensor):
            rgb = rgb.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported type {type(rgb)} for rgb")
        return rgb  # , done: deprecated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        obs, info = self.env.reset(seed=seed, options=options)
        rgb = info["rgb"]
        rgb = self.transpose_rgb(rgb)
        return rgb, info
