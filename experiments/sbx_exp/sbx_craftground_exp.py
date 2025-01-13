from craftground.screen_encoding_modes import ScreenEncodingMode
from experiments.get_device import get_device
from experiments.make_craftground_env import make_craftground_env
from experiments.experiment_setting import MAX_STEPS
from experiments.optim_dummy_vec_env import (
    DummyTensorVecEnv,
    patched_obs_as_tensor,
    TensorRolloutBuffer,
)
from experiments.transpose_vision_wrapper import TransposeVisionWrapper
from experiments.tree_wrapper import TreeWrapper
from experiments.cpu_wrapper import CPUVisionWrapper
from craftground.wrappers.vision import VisionWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from stable_baselines3.common import on_policy_algorithm
from wandb.integration.sb3 import WandbCallback

from sbx import PPO


def sbx_ppo_check(
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
        "MlpPolicy",  # Cnn
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

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for _ in range(1000):
    #     vec_env.render()
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)

    # vec_env.close()
