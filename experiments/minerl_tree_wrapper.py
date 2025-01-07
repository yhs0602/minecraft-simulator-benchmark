import gymnasium
from typing import SupportsFloat, Any
from gymnasium.core import WrapperActType, WrapperObsType
from shimmy import GymV21CompatibilityV0


class MineRLTreeWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        env = GymV21CompatibilityV0(env_id=None, make_kwargs=None, env=env)
        super().__init__(env)
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [
                2,  # forward
                2,  # back
                2,  # left
                2,  # right
                2,  # jump
                2,  # sneak
                2,  # sprint
                2,  # attack
                25,  # pitch
                25,  # yaw
            ]
        )
        self.observation_space = self.env.observation_space["pov"]

    def step(
        self, input_action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = {}
        # print(f"{action=}")
        action["forward"] = input_action[0]
        action["back"] = input_action[1]
        action["left"] = input_action[2]
        action["right"] = input_action[3]
        action["jump"] = input_action[4]
        action["sneak"] = input_action[5]
        action["sprint"] = input_action[6]
        action["attack"] = input_action[7]
        action["camera"] = [
            (input_action[8] - 12) * 15,
            (input_action[9] - 12) * 15,
        ]  # Pitch, yaw
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs["pov"]
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return obs["pov"], info
