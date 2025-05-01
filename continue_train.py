"""
Resume PPO training from an existing checkpoint.

Steps performed
---------------
1. Build a fresh monitored Drone2DEnv and wrap it in DummyVecEnv.
2. Reload the saved VecNormalize statistics so observation/
   reward scaling stays consistent with the original run.
3. Load the last policy weights and attach them to the env.
4. Continue learning for another 150 k timesteps
   (`reset_num_timesteps=False` keeps the global counter).
5. Save the new weights and the updated normaliser.

Run:
    python continue_train.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from drone_2d_env import Drone2DEnv


def make_env():
    """Factory returning a monitored env for the vector wrapper."""
    return Monitor(Drone2DEnv(max_steps=300), filename="train_logs/monitor_cont.csv")


def main() -> None:
    # 1) construct a single-process VecEnv
    env = DummyVecEnv([make_env])

    # 2) restore running mean/var so scaling matches previous training
    env = VecNormalize.load("train_logs/vec_normalize.pkl", env)
    env.training = True            # keep updating stats
    env.norm_reward = True

    # 3) load checkpoint weights
    model_path = "train_logs/ppo_drone_model.zip"
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Checkpoint not found. Train first or adjust path.")
    model = PPO.load(model_path, env=env)

    # 4) fine-tune for another 150 k steps
    model.learn(total_timesteps=150_000, reset_num_timesteps=False)

    # 5) save continued weights + updated normaliser
    model.save("train_logs/ppo_drone_model_continued.zip")
    env.save("train_logs/vec_normalize_continued.pkl")

    env.close()


if __name__ == "__main__":
    main()
