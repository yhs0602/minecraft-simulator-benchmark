# 1. Create a simple structure and save it as a .nbt file.
# 2. Load the .nbt file and put the agent on the structure.
# 3. Capture the depth image of the agent.
# 4. Save the depth image as a numpy array, and a png image.
import os
import craftground
from craftground.initial_environment_config import (
    DaylightMode,
    InitialEnvironmentConfig,
    ScreenEncodingMode,
    WorldType,
)
from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
from craftground.nbt.structure_editor import Structure
import cv2
import numpy as np

from experiments.check_vglrun import check_vglrun

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def create_nbt() -> str:
    out_path = os.path.join(THIS_DIR, "depth_structure.nbt")
    out_path = os.path.abspath(out_path)
    if os.path.exists(out_path):
        return out_path
    depth_structure = Structure()
    depth_structure.set_walls(0, 0, 0, 10, 10, 10, "minecraft:sea_lantern")
    depth_structure.save(out_path)
    return out_path


def create_environment(nbt_path: str):
    print(f"{nbt_path=}{(type(nbt_path))}")
    env = craftground.make(
        port=8000,
        initial_env_config=InitialEnvironmentConfig(
            image_width=640,
            image_height=360,
            hud_hidden=True,
            render_distance=11,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            requires_depth=True,
            world_type=WorldType.SUPERFLAT,
            requires_depth_conversion=True,
            structure_paths=[str(nbt_path)],
            initial_extra_commands=[
                "time set noon",
                f"place template minecraft:depth_structure 0 0 0",
                f"tp @p 2 2 2",
            ],
        ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        use_vglrun=check_vglrun(),
        verbose_python=True,
        verbose_gradle=True,
        verbose_jvm=True,
    )
    return env


def combine_depth_rgb(depth_frame, rgb_frame):
    """
    64x64 Grayscale Depth Frame 64x64 RGB Frame concat
    """

    # Concatenate Depth and RGB (64x64 → 64x128)
    # print(depth_bgr.shape, rgb_frame.shape)
    combined = np.hstack((depth_frame, rgb_frame))
    return combined


def depth_to_grayscale(depth: np.array) -> np.array:
    depth = depth / np.max(depth)
    return cv2.cvtColor(depth * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)


def depth_to_colormap(depth):
    """
    Convert depth values to a visually distinguishable colormap.
    """
    # Normalize depth to range 0-255
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # Apply a colormap (e.g., JET, VIRIDIS, or INFERNO)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    return depth_colormap


def overlay_depth_values(image, depth, sample_points):
    """
    Given an image and a depth map, overlay depth values at specific points.

    :param image: RGB or colormap depth image (H, W, 3)
    :param depth: Depth array (H, W)
    :param sample_points: List of (x, y) tuples indicating sample locations
    :return: Annotated image with depth values
    """
    annotated_image = image.copy()

    for x, y in sample_points:
        if (
            0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]
        ):  # Ensure points are within bounds
            depth_value = depth[y, x]
            text = f"{depth_value:.2f}"

            # Draw a small circle at the sample point
            cv2.circle(annotated_image, (x, y), 2, (255, 255, 255), -1)

            # Overlay text near the point
            cv2.putText(
                annotated_image,
                text,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return annotated_image


def main():
    nbt_path = create_nbt()
    env = create_environment(nbt_path=nbt_path)
    obs, info = env.reset()

    for i in range(20):
        action = no_op_v2()
        obs, reward, terminated, truncated, info = env.step(action)

    EPSILON = np.finfo(np.float16).eps

    height, width = 360, 640
    rows, cols = 18, 30
    sample_points = [
        (int(width * c / (cols - 1)), int(height * r / (rows - 1)))
        for r in range(rows)
        for c in range(cols)
    ]
    out_dir = os.path.join(THIS_DIR, "depth_test")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(10):
        action = no_op_v2()
        obs, reward, terminated, truncated, info = env.step(action)
        depth = np.asarray(info["full"].depth, dtype=np.float32).reshape(360, 640)
        # flip upside down depth
        depth = cv2.flip(depth, 0)

        # Save Depth as Numpy Array
        np.save(f"{out_dir}/depth_{i}.npy", depth)
        # Save depth as csv
        np.savetxt(f"{out_dir}/depth_{i}.csv", depth, delimiter=",")
        # Save Depth as PNG
        gray_depth = depth_to_grayscale(depth)
        rgb_depth = depth_to_colormap(depth)
        overlayed_rgb_depth = overlay_depth_values(rgb_depth, depth, sample_points)
        overlayed_gray_depth = overlay_depth_values(gray_depth, depth, sample_points)
        cv2.imwrite(f"{out_dir}/depth_gray_{i}.png", overlayed_gray_depth)
        cv2.imwrite(f"{out_dir}/depth_color_{i}.png", overlayed_rgb_depth)
        # Save RGB as PNG
        rgb_img = obs["pov"]
        overlayed_rgb_img = overlay_depth_values(rgb_img, depth, sample_points)
        cv2.imwrite(f"{out_dir}/rgb_{i}.png", overlayed_rgb_img)
        # Save Combined as PNG
        gray_combined = combine_depth_rgb(overlayed_gray_depth, overlayed_rgb_img)
        cv2.imwrite(f"{out_dir}/combined_gray_{i}.png", gray_combined)
        rgb_combined = combine_depth_rgb(overlayed_rgb_depth, overlayed_rgb_img)
        cv2.imwrite(f"{out_dir}/combined_color_{i}.png", rgb_combined)
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
