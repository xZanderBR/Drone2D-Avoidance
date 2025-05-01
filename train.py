"""
Train a PPO agent on the Drone-2D environment.

Features
--------
• Two separate environments:
      – training  (with observation / reward normalisation)
      – evaluation (normalisation *frozen* so scores are comparable)
• `Monitor` logs raw episode returns to CSV for quick plotting.
• `VecNormalize` keeps observations zero-mean / unit-var and
  scales rewards—crucial for PPO stability.
• `EvalCallback` evaluates every 10 k steps and stores the
  best-performing checkpoint in `train_logs/best_model.zip`.
• TensorBoard summaries are written to `ppo_drone_tb/`.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor           # per-episode CSV logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from drone_2d_env import Drone2DEnv


# ───────────────────────── helper factories ──────────────────────────
def make_train_env():
    """
    Returns a *single* monitored Drone2DEnv for the training vector-env.
    max_steps=300 keeps episodes short; no rendering during training.
    """
    return Monitor(Drone2DEnv(max_steps=300), filename="train_logs/monitor_train.csv")


def make_eval_env():
    """
    Factory for the evaluation environment.
    Separate CSV so eval metrics don’t mix with training stats.
    """
    return Monitor(Drone2DEnv(max_steps=300), filename="eval_logs/monitor_eval.csv")


# ───────────────────────────── main script ───────────────────────────
def main() -> None:
    os.makedirs("train_logs", exist_ok=True)
    os.makedirs("eval_logs",  exist_ok=True)

    # 1) build vectorised envs ------------------------------------------------
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=True,  norm_reward=True)

    eval_env  = DummyVecEnv([make_eval_env])
    # freeze running stats so eval rewards are on the original scale
    eval_env  = VecNormalize(eval_env,  norm_obs=True,  norm_reward=False, training=False)

    # 2) instantiate PPO ------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0,
        clip_range=0.20,
        verbose=1,
        tensorboard_log="ppo_drone_tb",
        seed=0,
    )

    # 3) periodic evaluation callback ----------------------------------------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="train_logs",  # writes best_model.zip
        log_path="eval_logs",
        eval_freq=10_000,                   # every 10k environment steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # 4) training loop --------------------------------------------------------
    model.learn(total_timesteps=1_500_000, callback=eval_cb)

    # 5) final save -----------------------------------------------------------
    model.save("train_logs/ppo_drone_model")          # final weights
    train_env.save("train_logs/vec_normalize.pkl")    # running mean/var

    # clean up pygame / vecenv resources
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
