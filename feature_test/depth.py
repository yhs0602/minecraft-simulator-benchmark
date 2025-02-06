import craftground
from craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.initial_environment_config import (
    DaylightMode,
    InitialEnvironmentConfig,
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
    verbose_gradle: bool = False,
    verbose_jvm: bool = False,
    use_shmem: bool = False,
    requires_depth_conversion: bool = False,
) -> CraftGroundEnvironment:
    return craftground.make(
        port=port,
        initial_env_config=InitialEnvironmentConfig(
            image_width=width,
            image_height=height,
            hud_hidden=False,
            render_distance=11,
            screen_encoding_mode=screen_encoding_mode,
            requires_depth=True,
            requires_depth_conversion=requires_depth_conversion,
        ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        use_vglrun=check_vglrun(),
        verbose_python=verbose_python,
        verbose_gradle=verbose_gradle,
        verbose_jvm=verbose_jvm,
        # use_shared_memory=use_shmem,
    )


def draw_stats(all_depths, conversion: bool):
    mean_depth = [np.mean(depth) for depth in all_depths]
    std_depth = [np.std(depth) for depth in all_depths]
    min_depth = [np.min(depth) for depth in all_depths]
    max_depth = [np.max(depth) for depth in all_depths]

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(mean_depth)
    plt.title("Mean depth")
    plt.subplot(2, 2, 2)
    plt.plot(std_depth)
    plt.title("Std depth")
    plt.subplot(2, 2, 3)
    plt.plot(min_depth)
    plt.title("Min depth")
    plt.subplot(2, 2, 4)
    plt.plot(max_depth)
    plt.title("Max depth")
    plt.show()

    concatenated_depth = np.concatenate(all_depths).ravel()
    plt.hist(concatenated_depth, bins=100, color="blue", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Depth values")
    plt.ylabel("Frequency")
    plt.title(
        f"Histogram of depth values total({concatenated_depth.size})(conversion={conversion})"
    )
    plt.show()


def depth_to_grayscale(depth):
    # depth = depth / np.max(depth)
    return (depth * 255).astype(np.uint8)


def combine_depth_rgb(depth_frame, rgb_frame):
    """
    64x64 Grayscale Depth Frame 64x64 RGB Frame concat
    """
    depth_bgr = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
    # flip upside down depth frame
    depth_bgr = cv2.flip(depth_bgr, 0)

    # Concatenate Depth and RGB (64x64 → 64x128)
    # print(depth_bgr.shape, rgb_frame.shape)
    combined = np.hstack((depth_bgr, rgb_frame))
    return combined


def save_depth_video(depth_frames, rgb_frames, filename="depth_video.mp4", fps=30):
    """
    Save Grayscale image list as a video
    :param frames: Grayscale image list (np.uint8, shape=(64,64))
    :param filename: file name to save
    :param fps: frames per second (default 30)
    """
    concatenated_frames = [
        combine_depth_rgb(depth, rgb) for depth, rgb in zip(depth_frames, rgb_frames)
    ]

    height, width, chn = concatenated_frames[0].shape  # 64x128 resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)

    for frame in concatenated_frames:
        out.write(frame)  # Grayscale → BGR
    out.release()


if __name__ == "__main__":
    env = make_craftground_env(
        port=8000,
        width=64,
        height=64,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        verbose_python=False,
        verbose_gradle=True,
        verbose_jvm=True,
        use_shmem=False,
        requires_depth_conversion=False,
    )

    obs, info = env.reset()

    EPSILON = np.finfo(np.float16).eps

    all_depths = []
    rgb_frames = []
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        depth = np.asarray(info["full"].depth, dtype=np.float32).reshape(64, 64)
        assert depth.shape == (64, 64)
        # raw depth values should be in the range [0, 1]
        if not np.all(depth >= -EPSILON):
            print(depth[depth < -EPSILON])
            raise ValueError("Depth values should be non-negative")
        if not np.all(depth <= 1 + EPSILON):
            print(depth[depth > 1 + EPSILON])
            raise ValueError("Depth values should be less than or equal to 1")
        all_depths.append(depth)
        rgb_frames.append(obs["pov"])
        if terminated or truncated:
            break
    env.close()

    print("Depth values are in the range [0, 1]!!")

    # draw some statistics
    draw_stats(all_depths, conversion=False)

    # save depth video
    save_depth_video(
        [depth_to_grayscale(depth) for depth in all_depths],
        rgb_frames,
        filename="depth_video_raw.mp4",
    )

    all_depths = []
    rgb_frames = []
    env = make_craftground_env(
        port=8000,
        width=64,
        height=64,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        verbose_python=False,
        verbose_gradle=False,
        verbose_jvm=False,
        use_shmem=False,
        requires_depth_conversion=True,
    )

    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        depth = np.asarray(info["full"].depth, dtype=np.float32).reshape(64, 64)
        assert depth.shape == (64, 64)
        # converted depth values should be in the range [0, 1]
        # if not np.all(depth >= -EPSILON):
        #     print(depth[depth < -EPSILON])
        #     raise ValueError("Depth values should be non-negative")
        # if not np.all(depth <= 1 + EPSILON):
        #     print(depth[depth > 1 + EPSILON])
        #     raise ValueError("Depth values should be less than or equal to 1")
        all_depths.append(depth)
        rgb_frames.append(obs["pov"])
        if terminated or truncated:
            break
    print("Depth values are in the range [0, 1]!!")
    env.close()

    # draw some statistics
    draw_stats(all_depths, conversion=True)
    # save depth video
    save_depth_video(
        [depth_to_grayscale(depth) for depth in all_depths],
        rgb_frames,
        filename="depth_video_converted.mp4",
    )
