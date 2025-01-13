from craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.initial_environment_config import DaylightMode
import gymnasium as gym
import craftground
from craftground import InitialEnvironmentConfig, ActionSpaceVersion

from check_vglrun import check_vglrun


def make_craftground_env(
    port: int,
    width: int,
    height: int,
    screen_encoding_mode: ScreenEncodingMode,
    verbose_python: bool = False,
    verbose_gradle: bool = True,
    verbose_jvm: bool = True,
) -> gym.Env:
    return craftground.make(
        port=port,
        initial_env_config=InitialEnvironmentConfig(
            image_width=width,
            image_height=height,
            hud_hidden=False,
            render_distance=11,
            screen_encoding_mode=screen_encoding_mode,
        ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        use_vglrun=check_vglrun(),
        verbose_python=verbose_python,
        verbose_gradle=verbose_gradle,
        verbose_jvm=verbose_jvm,
    )
