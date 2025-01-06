import argparse
import logging
import sys
import time
import wandb
import gym
import minerl  # noqa


from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List
from minerl.herobraine.hero import handlers as H
from minerl.herobraine.env_specs.human_controls import SimpleHumanEmbodimentEnvSpec
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from experiments.experiment_setting import MAX_STEPS

# logging.basicConfig(level=logging.DEBUG)


# https://minerl.readthedocs.io/en/v0.4.4/tutorials/custom_environments.html
class MLGWB(SimpleHumanEmbodimentEnvSpec):
    def __init__(self, width, height, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "MLGWB-v0"

        kwargs["resolution"] = (width, height)

        super().__init__(
            *args, max_episode_steps=MAX_STEPS, reward_threshold=100.0, **kwargs
        )

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_rewardables(self) -> List[Handler]:
        return []

    def create_server_quit_producers(self):
        return [handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_server_decorators(self) -> List[Handler]:
        return []

    # the episode can terminate when this is True
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return False

    def is_from_folder(self, folder: str) -> bool:
        return folder == "mlgwb"

    def get_docstring(self):
        return "MLGWB_DOC"

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            # Sets time to morning and stops passing of time
            handlers.TimeInitialCondition(False, 23000)
        ]

    def create_server_world_generators(self) -> List[Handler]:
        return [handlers.DefaultWorldGenerator(force_reset=False)]


def make_minerl_env(
    width: int,
    height: int,
):
    abs_MLG = MLGWB(width=width, height=height)
    abs_MLG.register()
    env = gym.make("MLGWB-v0")

    return env


# Simulation noop + render using moviepy, not optimized
def render_check(
    run,
    vision_width: int,
    vision_height: int,
):
    env = make_minerl_env(
        width=vision_width,
        height=vision_height,
    )
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
        for i in range(MAX_STEPS):
            action = [env.action_space.noop()]
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
    vision_width: int,
    vision_height: int,
):
    env = make_minerl_env(
        width=vision_width,
        height=vision_height,
    )

    obs = env.reset()  # MineRL uses old gym interface
    done = False
    start_time = time.time_ns()
    for i in range(MAX_STEPS):
        action = env.action_space.noop()
        obs, reward, done, _ = env.step(action)
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


def do_experiment(image_width, load):
    vision_width, vision_height = {
        "64x64": (64, 64),
        "114x64": (114, 64),
        "640x360": (640, 360),
    }[image_width]

    group_name = f"minerl100--{vision_width}-{vision_height}-{load}"
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
        simulation_check(vision_width, vision_height)
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

    args = parser.parse_args()

    # Display the selected configuration
    print(f"Running experiment with the following settings:")
    print(f"Image Width: {args.image_width}")
    print(f"Load: {args.load}")
    do_experiment(args.image_width, args.load)


if __name__ == "__main__":
    main()
