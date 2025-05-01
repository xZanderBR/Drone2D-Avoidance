# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.monitor   import Monitor
from stable_baselines3.common.vec_env   import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from drone_2d_env import Drone2DEnv
import os

def make_train_env():
    return Monitor(Drone2DEnv(max_steps=300), "train_logs/monitor_train.csv")

def make_eval_env():
    return Monitor(Drone2DEnv(max_steps=300), "eval_logs/monitor_eval.csv")
 
def main():
    os.makedirs("train_logs", exist_ok=True)
    os.makedirs("eval_logs",  exist_ok=True)

    # ── vectorized envs ──
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env  = DummyVecEnv([make_eval_env])
    eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False, training=False)

    # ── PPO ──
    model = PPO("MlpPolicy",
                train_env,
                learning_rate=5e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                ent_coef=0.01,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="ppo_drone_tb",
                seed=0)

    # ── periodic evaluation ──
    eval_cb = EvalCallback(eval_env,
                           best_model_save_path="train_logs",
                           log_path="eval_logs",
                           eval_freq=10_000,
                           n_eval_episodes=5,
                           deterministic=True,
                           render=False)

    model.learn(total_timesteps=1_500_000, callback=eval_cb)

    # ── save ──
    model.save("train_logs/ppo_drone_model")
    train_env.save("train_logs/vec_normalize.pkl")

    train_env.close(); eval_env.close()


if __name__ == "__main__":
    main()
