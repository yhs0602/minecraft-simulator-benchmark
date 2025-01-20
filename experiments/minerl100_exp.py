import argparse
import platform
import sys
import time
import jax
import wandb
import gym
import minerl  # noqa


from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List
from minerl.herobraine.env_specs.human_controls import SimpleHumanEmbodimentEnvSpec
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from sbx import PPO as JAXPPO
from wandb.integration.sb3 import WandbCallback
from experiments.minerl_tree_wrapper import MineRLTreeWrapper
from get_device import get_device

from experiments.experiment_setting import MAX_STEPS

# logging.basicConfig(level=logging.DEBUG)


# https://minerl.readthedocs.io/en/v0.4.4/tutorials/custom_environments.html
class MLGWB(SimpleHumanEmbodimentEnvSpec):
    def __init__(self, width, height, render_mode, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "MLGWB-v0"

        kwargs["resolution"] = (width, height)
        self.render_mode = render_mode

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
    abs_MLG = MLGWB(width=width, height=height, render_mode="rgb_array")
    abs_MLG.register()
    env = gym.make("MLGWB-v0")
    env.render_mode = "rgb_array"

    return env


def sbx_ppo_check(
    run,
    vision_width: int,
    vision_height: int,
    device: str,
    render: bool = False,
    use_optimized_sb3: bool = False,
    max_steps: int = MAX_STEPS,
):
    base_env = make_minerl_env(
        width=vision_width,
        height=vision_height,
    )
    base_env = MineRLTreeWrapper(base_env)
    env = DummyVecEnv([lambda: base_env])
    env.render_mode = "rgb_array"
    if render:
        # Record video every 2000 steps and save the video
        env = VecVideoRecorder(
            env,
            f"videos/{run.id}",
            record_video_trigger=lambda x: x % 2000 == 0,
            video_length=2000,
        )
    model = JAXPPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
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
        # model.save(f"ckpts/{run.group}-{run.name}.ckpt")
    finally:
        env.close()
        run.finish()


def ppo_check(
    run,
    vision_width: int,
    vision_height: int,
    device_id: int = 3,
    render: bool = False,
    use_optimized_sb3: bool = False,
    max_steps: int = MAX_STEPS,
):
    base_env = make_minerl_env(
        width=vision_width,
        height=vision_height,
    )
    base_env = MineRLTreeWrapper(base_env)
    env = DummyVecEnv([lambda: base_env])
    env.render_mode = "rgb_array"
    if render:
        # Record video every 2000 steps and save the video
        env = VecVideoRecorder(
            env,
            f"videos/{run.id}",
            record_video_trigger=lambda x: x % 2000 == 0,
            video_length=2000,
        )
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
        # model.save(f"ckpts/{run.group}-{run.name}.ckpt")
    finally:
        env.close()
        run.finish()


# Simulation noop + render using moviepy, not optimized
def render_check(
    run,
    vision_width: int,
    vision_height: int,
):
    base_env = make_minerl_env(
        width=vision_width,
        height=vision_height,
    )
    base_env.render_mode = "rgb_array"
    # To record videos, we need to wrap the environment with VecVideoRecorder
    env = DummyVecEnv([lambda: base_env])
    env.render_mode = "rgb_array"
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
            action = [base_env.action_space.noop()]
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


def do_experiment(image_width, load, device: str, max_steps: int):
    vision_width, vision_height = {
        "64x64": (64, 64),
        "114x64": (114, 64),
        "640x360": (640, 360),
    }[image_width]

    group_name = "v2-"
    if device == "cpu":
        group_name += "cpu-"
        jax.config.update("jax_platform_name", "cpu")
    else:
        group_name += ""
    if platform.system() == "Darwin":
        group_name += f"minerl100-apple--{vision_width}-{vision_height}-{load}"
        print("Running on macOS")
    else:
        group_name += f"minerl100--{vision_width}-{vision_height}-{load}"
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
        simulation_check(vision_width, vision_height)
    elif load == "render":
        render_check(run, vision_width, vision_height)
    elif load == "ppo":
        ppo_check(run, vision_width, vision_height, render=False)
    elif load == "render_ppo":
        ppo_check(run, vision_width, vision_height, render=True)
    elif load == "optimized_render":
        pass
    elif load == "optimized_ppo":
        pass
    elif load == "optimized_render_ppo":
        pass
    elif load == "sbx-ppo":
        sbx_ppo_check(
            run,
            vision_width,
            vision_height,
            render=False,
            use_optimized_sb3=False,
            max_steps=max_steps,
            device=device,
        )
    elif load == "render_sbx-ppo":
        sbx_ppo_check(
            run,
            vision_width,
            vision_height,
            render=True,
            use_optimized_sb3=False,
            max_steps=max_steps,
            device=device,
        )
    elif load == "optimized_sbx-ppo":
        sbx_ppo_check(
            run,
            vision_width,
            vision_height,
            render=False,
            use_optimized_sb3=True,
            max_steps=max_steps,
            device=device,
        )
    elif load == "optimized_render_sbx-ppo":
        sbx_ppo_check(
            run,
            vision_width,
            vision_height,
            render=True,
            use_optimized_sb3=True,
            max_steps=max_steps,
            device=device,
        )
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
            "sbx-ppo",
            "render_sbx-ppo",
            "optimized_sbx-ppo",
            "optimized_render_sbx-ppo",
        ],
        required=True,
        help="Specify the load configuration.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=get_device(3),
        help="Device to use for the experiment.",
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
    do_experiment(args.image_width, args.load, args.device, args.max_steps)


if __name__ == "__main__":
    main()
