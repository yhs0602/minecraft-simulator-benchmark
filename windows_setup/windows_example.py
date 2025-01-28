import craftground
from stable_baselines3 import A2C

from windows_setup.vision_wrapper import VisionWrapper

# Initialize environment
if __name__ == "__main__":
    env = craftground.make(port=8023)
    env = VisionWrapper(env, 640, 360)

    # Train model
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("a2c_craftground")
