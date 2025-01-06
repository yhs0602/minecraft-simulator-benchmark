import argparse
import sys
import time
from craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.initial_environment_config import DaylightMode
import wandb
import gymnasium as gym
import craftground
from craftground import InitialEnvironmentConfig, ActionSpaceVersion
from craftground.wrappers.vision import VisionWrapper
from craftground.minecraft import no_op_v2

from check_vglrun import check_vglrun

MAX_STEPS = 100_000


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


def simulation_check(
    screen_encoding_mode: ScreenEncodingMode,
    vision_width: int,
    vision_height: int,
    port: int,
):
    env = make_craftground_env(
        port=port,
        width=vision_width,
        height=vision_height,
        screen_encoding_mode=screen_encoding_mode,
    )
    VisionWrapper(env, x_dim=vision_width, y_dim=vision_height)
    obs, info = env.reset()  # info
    start_time = time.time_ns()
    for i in range(MAX_STEPS):
        action = no_op_v2()
        obs, reward, terminated, truncated, info = env.step(action)  # truncated
        time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
        fps = int(i / time_elapsed)
        if i % 512 == 0:
            wandb.log(
                {
                    "time/iterations": i,
                    "time/fps": fps,
                    "time/time_elapsed": int(time_elapsed),
                    "time/total_timesteps": i,
                }
            )
        if i % 4000 == 0:
            print(f"Step: {i}")


def do_experiment(mode, image_width, load, port):
    screen_encoding_mode = {
        "raw": ScreenEncodingMode.RAW,
        "zerocopy": ScreenEncodingMode.ZEROCOPY,
    }[mode]
    vision_width, vision_height = {
        "64x64": (64, 64),
        "114x64": (114, 64),
        "640x320": (640, 320),
    }[image_width]

    group_name = f"craftground-{mode}--{vision_width}-{vision_height}-{load}"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="minecraft-envs-performance-comparison",
        entity="jourhyang123",
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    if load == "simulation":
        simulation_check(screen_encoding_mode, vision_width, vision_height, port)
    elif load == "render":
        pass
    elif load == "ppo":
        pass
    elif load == "render_ppo":
        pass
    elif load == "optimized_render":
        pass
    elif load == "optimized_ppo":
        pass
    elif load == "optimized_render_ppo":
        pass
    else:
        raise ValueError(f"Unknown load configuration: {load}")


def main():
    parser = argparse.ArgumentParser(
        description="Run performance comparison experiments."
    )

    # Environment options
    parser.add_argument(
        "--mode",
        choices=["raw", "zerocopy"],
        required=True,
        help="Select the environment to run the experiment on.",
    )

    # Image width options
    parser.add_argument(
        "--image_width",
        choices=["64x64", "114x64", "640x320"],
        required=True,
        help="Set the resolution of the image.",
    )

    # Load options
    parser.add_argument(
        "--load",
        choices=[
            "simulation",
            "render",
            "ppo",
            "render_ppo",
            "optimized_render",
            "optimized_ppo",
            "optimized_render_ppo",
        ],
        required=True,
        help="Specify the load configuration.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port number to use for the environment.",
    )

    args = parser.parse_args()

    # Display the selected configuration
    print(f"Running experiment with the following settings:")
    print(f"Image Width: {args.image_width}")
    print(f"Load: {args.load}")
    print(f"Mode: {args.mode}")
    print(f"Port: {args.port}")

    do_experiment(args.mode, args.image_width, args.load, args.port)


if __name__ == "__main__":
    main()
