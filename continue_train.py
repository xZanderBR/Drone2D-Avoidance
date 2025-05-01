import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from drone_2d_env import Drone2DEnv

def make_env():
    return Monitor(Drone2DEnv(max_steps=300), "train_logs/monitor_cont.csv")

def main():
    # 1. Re-create the VecEnv
    env = DummyVecEnv([make_env])

    # 2. Reload saved VecNormalize statistics
    env = VecNormalize.load("train_logs/vec_normalize.pkl", env)
    env.training    = True          # keep updating stats
    env.norm_reward = True

    # 3. Load the existing PPO model and bind it to the env
    model_path = "train_logs/ppo_drone_model.zip"
    model = PPO.load(model_path, env=env)

    # 4. Continue learning (add another 300 k steps, for example)
    model.learn(total_timesteps=300_000,
                reset_num_timesteps=False)     # ← keep global timestep counter

    # 5. Save updated weights & VecNormalize
    model.save("train_logs/ppo_drone_model_continued")
    env.save("train_logs/vec_normalize_continued.pkl")

    env.close()

if __name__ == "__main__":
    main()