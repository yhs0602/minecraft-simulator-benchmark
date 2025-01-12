import argparse
import sys
import time
from craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.initial_environment_config import DaylightMode
import wandb
from experiments.optim_dummy_vec_env import (
    DummyTensorVecEnv,
    patched_obs_as_tensor,
    TensorRolloutBuffer,
)
from experiments.sbx.sbx_craftground_exp import sbx_ppo_check
from experiments.transpose_vision_wrapper import TransposeVisionWrapper
from experiments.tree_wrapper import TreeWrapper
from get_device import get_device
from experiments.cpu_wrapper import CPUVisionWrapper
from experiments.experiment_setting import MAX_STEPS
import gymnasium as gym
import craftground
from craftground import InitialEnvironmentConfig, ActionSpaceVersion
from craftground.wrappers.vision import VisionWrapper
from craftground.minecraft import no_op_v2
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback

from check_vglrun import check_vglrun


from stable_baselines3.common import on_policy_algorithm
import platform


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


def ppo_check(
    run,
    screen_encoding_mode: ScreenEncodingMode,
    vision_width: int,
    vision_height: int,
    port: int,
    device_id: int = 3,
    render: bool = False,
    use_optimized_sb3: bool = False,
    max_steps: int = MAX_STEPS,
):
    env = make_craftground_env(
        port=port,
        width=vision_width,
        height=vision_height,
        screen_encoding_mode=screen_encoding_mode,
    )
    env = VisionWrapper(env, x_dim=vision_width, y_dim=vision_height)
    env = TreeWrapper(env)

    # Record video every 2000 steps and save the video
    if render or not use_optimized_sb3:
        if screen_encoding_mode == ScreenEncodingMode.ZEROCOPY:
            env = CPUVisionWrapper(env)
        env = DummyVecEnv([lambda: env])
        if render:
            env = VecVideoRecorder(
                env,
                f"videos/{run.id}",
                record_video_trigger=lambda x: x % 2000 == 0,
                video_length=2000,
            )
    else:
        env = TransposeVisionWrapper(env, x_dim=vision_width, y_dim=vision_height)
        env = DummyTensorVecEnv([lambda: env])
        on_policy_algorithm.obs_as_tensor = patched_obs_as_tensor
        on_policy_algorithm.RolloutBuffer = TensorRolloutBuffer
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=get_device(device_id),
        tensorboard_log=f"runs/{run.id}",
        gae_lambda=0.99,
        ent_coef=0.005,
        n_steps=512,
    )

    try:
        env.reset()
        model.learn(
            total_timesteps=max_steps,
            callback=[
                # CustomWandbCallback(),
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
            ],
        )
        model.save(f"ckpts/{run.group}-{run.name}.ckpt")
    finally:
        env.close()
        run.finish()


# Simulation noop + render using moviepy, not optimized
def render_check(
    run,
    screen_encoding_mode: ScreenEncodingMode,
    vision_width: int,
    vision_height: int,
    port: int,
    optimize: bool = False,
    max_steps: int = MAX_STEPS,
):
    env = make_craftground_env(
        port=port,
        width=vision_width,
        height=vision_height,
        screen_encoding_mode=screen_encoding_mode,
    )
    env = VisionWrapper(env, x_dim=vision_width, y_dim=vision_height)
    if screen_encoding_mode == ScreenEncodingMode.ZEROCOPY and not optimize:
        env = CPUVisionWrapper(env)

    # To record videos, we need to wrap the environment with VecVideoRecorder
    env = DummyVecEnv([lambda: env])
    # Record video every 2000 steps and save the video
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=2000,
    )
    try:
        obs = env.reset()  # DummyVecEnv reset returns only obs
        start_time = time.time_ns()
        for i in range(max_steps):
            action = [no_op_v2()]
            obs, reward, terminated, info = env.step(
                action
            )  # truncated is not provided in VecEnv
            time_elapsed = max(
                (time.time_ns() - start_time) / 1e9, sys.float_info.epsilon
            )
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
    finally:
        env.close()
        run.finish()


def simulation_check(
    screen_encoding_mode: ScreenEncodingMode,
    vision_width: int,
    vision_height: int,
    port: int,
    max_steps: int,
):
    env = make_craftground_env(
        port=port,
        width=vision_width,
        height=vision_height,
        screen_encoding_mode=screen_encoding_mode,
    )
    obs, info = env.reset()  # info
    start_time = time.time_ns()
    for i in range(max_steps):
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

    env.terminate()


def do_experiment(mode, image_width, load, port, max_steps: int):
    screen_encoding_mode = {
        "raw": ScreenEncodingMode.RAW,
        "zerocopy": ScreenEncodingMode.ZEROCOPY,
    }[mode]
    vision_width, vision_height = {
        "64x64": (64, 64),
        "114x64": (114, 64),
        "640x360": (640, 360),
    }[image_width]

    if platform.system() == "Darwin":
        group_name = f"craftground-apple-{mode}--{vision_width}-{vision_height}-{load}"
        print("Running on macOS")
    else:
        group_name = f"craftground-{mode}--{vision_width}-{vision_height}-{load}"
    print(f"Group name: {group_name}")
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
        simulation_check(
            screen_encoding_mode, vision_width, vision_height, port, max_steps
        )
    elif load == "render":
        render_check(
            run, screen_encoding_mode, vision_width, vision_height, port, max_steps
        )
    elif load == "ppo":
        ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=False,
            max_steps=max_steps,
        )
    elif load == "render_ppo":
        ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=True,
            max_steps=max_steps,
        )
    elif load == "optimized_render":
        render_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            optimize=True,
            max_steps=max_steps,
        )
    elif load == "optimized_ppo":
        ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=False,
            optimize=True,
            max_steps=max_steps,
        )
    elif load == "optimized_render_ppo":
        ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=True,
            optimize=True,
            max_steps=max_steps,
        )
    elif load == "sbx-ppo":
        sbx_ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=True,
            optimize=True,
            max_steps=max_steps,
        )
    elif load == "render_sbx-ppo":
        sbx_ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=True,
            optimize=True,
            max_steps=max_steps,
        )
    elif load == "optimized_sbx-ppo":
        sbx_ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=True,
            optimize=True,
            max_steps=max_steps,
        )
    elif load == "optimized_render_sbx-ppo":
        sbx_ppo_check(
            run,
            screen_encoding_mode,
            vision_width,
            vision_height,
            port,
            render=True,
            optimize=True,
            max_steps=max_steps,
        )
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
        choices=["64x64", "114x64", "640x360"],
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

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum number of steps to run the experiment.",
    )

    args = parser.parse_args()

    # Display the selected configuration
    print(f"Running experiment with the following settings:")
    print(f"Image Width: {args.image_width}")
    print(f"Load: {args.load}")
    print(f"Mode: {args.mode}")
    print(f"Port: {args.port}")
    print(f"Max Steps: {args.max_steps}")

    do_experiment(args.mode, args.image_width, args.load, args.port, args.max_steps)


if __name__ == "__main__":
    main()
