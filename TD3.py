import os
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Custom Callback to Track and Plot
class RewardPlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
                    self.ep_lengths.append(info["episode"]["l"])
        return True

    def plot(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.ep_rewards, label="ep_rew_mean")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Reward")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.ep_lengths, label="ep_len_mean", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode Length")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Use CPU
device = "cpu"
print(f"Using device: {device.upper()}")

# Environment Setup
env_id = "Ant-v5"
train_env = make_vec_env(env_id, n_envs=1)

n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=torch.zeros(n_actions),
    sigma=0.1 * torch.ones(n_actions)
)

# Model Setup
model = TD3(
    "MlpPolicy",
    train_env,
    action_noise=action_noise,
    verbose=1,
    device=device,
    batch_size=256,
    learning_rate=1e-4,
    tau=0.01,
    buffer_size=120_000,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    policy_kwargs=dict(net_arch=[400, 300])
)

# Training
reward_callback = RewardPlotCallback()
model.learn(total_timesteps=120_000, callback=reward_callback)
model.save("td3_ant_v5")
print("Model trained and saved as 'td3_ant_v5.zip'.")

# Plotting
reward_callback.plot()

# Evaluation and Video
eval_env = gym.make(env_id, render_mode="rgb_array")
obs, info = eval_env.reset()
frames = []
total_reward = 0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward

    frames.append(eval_env.render())

    if terminated or truncated:
        print(f"Episode ended early at step {step}, reward: {total_reward:.2f}")
        obs, info = eval_env.reset()
        total_reward = 0

eval_env.close()

# Save Video
video_path = "td3_ant_eval.mp4"
iio.imwrite(video_path, np.array(frames), fps=30)
print(f"Evaluation video saved as '{video_path}'")

