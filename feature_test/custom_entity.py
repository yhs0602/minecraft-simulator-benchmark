import craftground
from craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.initial_environment_config import (
    DaylightMode,
    InitialEnvironmentConfig,
    WorldType,
)
from craftground.environment.action_space import ActionSpaceVersion
from craftground import CraftGroundEnvironment
from matplotlib import pyplot as plt
import numpy as np
import cv2

from experiments.check_vglrun import check_vglrun


def make_craftground_env(
    port: int,
    width: int,
    height: int,
    screen_encoding_mode: ScreenEncodingMode,
    verbose_python: bool = False,
    verbose_gradle: bool = True,
    verbose_jvm: bool = True,
    use_shmem: bool = False,
) -> CraftGroundEnvironment:
    return craftground.make(
        port=port,
        initial_env_config=InitialEnvironmentConfig(
            image_width=width,
            image_height=height,
            hud_hidden=True,
            render_distance=11,
            screen_encoding_mode=screen_encoding_mode,
            initial_extra_commands=[
                "summon entitytesting:realistic_human 1 -59 1",
                "summon entitytesting:realistic_human -1 -59 1",
                "summon entitytesting:realistic_human 1 -59 -1",
                "summon entitytesting:realistic_human -1 -59 -1",
            ],
            world_type=WorldType.SUPERFLAT,
        ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY)
        .set_initial_position(0, -60.0, 0, 0, 0),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        use_vglrun=check_vglrun(),
        verbose_python=verbose_python,
        verbose_gradle=verbose_gradle,
        verbose_jvm=verbose_jvm,
        # use_shared_memory=use_shmem,
    )


if __name__ == "__main__":
    env = make_craftground_env(
        port=8000,
        width=640,
        height=320,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        verbose_python=False,
        verbose_gradle=True,
        verbose_jvm=True,
        use_shmem=False,
    )

    obs, info = env.reset()

    EPSILON = np.finfo(np.float16).eps

    rgb_frames = []
    for _ in range(1000):
        action = env.action_space.sample()
        action['inventory'] = 0
        action['drop'] = 0
        action['forward'] = 0
        action['back'] = 0
        action['jump'] = 0
        action['left'] = 0
        action['right'] = 0
        action['sneak'] = 0
        action['sprint'] = 0
        action['attack'] = 0
        action['use'] = 0
        action['camera'][0] = np.clip(action['camera'][0], -45, 45)
        action['camera'][1] = np.clip(action['camera'][1], -45, 45)
        obs, reward, terminated, truncated, info = env.step(action)
        # info["full"]
        rgb_frames.append(obs["pov"])
        if terminated or truncated:
            break
    env.close()
    rgb_frames = np.array(rgb_frames)

    # 비디오 저장 경로
    video_path = "craftground_output.mp4"

    # 프레임 크기 (가정: obs["pov"]의 shape는 (H, W, 3))
    frame_height, frame_width = rgb_frames[0].shape[:2]

    # 비디오 저장 객체 생성
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,  # FPS
        (frame_width, frame_height),
    )

    # 프레임 쓰기 (OpenCV는 BGR을 사용하므로 RGB를 BGR로 변환)
    for frame in rgb_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"비디오가 '{video_path}'로 저장되었습니다.")
