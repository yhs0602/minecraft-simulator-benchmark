import argparse
import gymnasium as gym
import argparse
import sys
import time
from craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.initial_environment_config import DaylightMode
import wandb
from experiments.craftground_exp import make_craftground_env
from experiments.optim_dummy_vec_env import (
    DummyTensorVecEnv,
    patched_obs_as_tensor,
    TensorRolloutBuffer,
)
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
from wandb.integration.sb3 import WandbCallback

from check_vglrun import check_vglrun


from stable_baselines3.common import on_policy_algorithm
import platform


from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ


def sbx_ppo_check(mode, image_width, load, port, max_steps: int):
    screen_encoding_mode = {
        "raw": ScreenEncodingMode.RAW,
        "zerocopy": ScreenEncodingMode.ZEROCOPY,
    }[mode]
    vision_width, vision_height = {
        "64x64": (64, 64),
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

    env = make_craftground_env(
        port=port,
        width=vision_width,
        height=vision_height,
        screen_encoding_mode=screen_encoding_mode,
    )
    env = VisionWrapper(env, x_dim=vision_width, y_dim=vision_height)
    env = TreeWrapper(env)

    # env = gym.make("Pendulum-v1", render_mode="human")
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

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=max_steps, progress_bar=True)

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for _ in range(1000):
    #     vec_env.render()
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)

    # vec_env.close()


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
            "sbx-ppo",
            "render_sbx-ppo",
            "optimized_sbx-ppo",
            "optimized_render_sbx-ppo",
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