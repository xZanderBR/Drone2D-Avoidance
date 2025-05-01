"""
Render and evaluate a trained PPO agent in the Drone-2D
environment.

Example
-------
    python eval.py                 # 25 episodes, ~50 FPS
    python eval.py -e 5 -d 0.05    # 5 episodes, 20 FPS
    python eval.py --model train_logs/best_model.zip \
                   --norm  train_logs/vec_normalize.pkl

Episode outcome (`event`) will be one of:
goal · collision · stuck · slow · timeout
"""

import argparse
import os
import time
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_2d_env import Drone2DEnv


# ───────────────────────── helper ──────────────────────────
def unwrap(val: Any):
    """Return scalar from 1-element container, else the value unchanged."""
    return val[0] if isinstance(val, (np.ndarray, list, tuple)) else val


# ───────────────────────── main ────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=25, help="number of episodes to run")
    parser.add_argument("-d", "--delay", type=float, default=0.02, help="seconds between rendered frames")
    parser.add_argument("--model", default="train_logs/ppo_drone_model.zip", help="path to .zip weights")
    parser.add_argument("--norm",  default="train_logs/vec_normalize.pkl",   help="path to VecNormalize .pkl")
    args = parser.parse_args()

    # 1) build a single render-capable environment
    venv = DummyVecEnv([lambda: Drone2DEnv(render_mode="human")])

    # 2) restore normalisation statistics (frozen in eval mode)
    if not os.path.isfile(args.norm):
        raise FileNotFoundError(f"VecNormalize stats not found at {args.norm}")
    venv = VecNormalize.load(args.norm, venv)
    venv.training = False
    venv.norm_reward = False

    # 3) load the PPO policy
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model}")
    model = PPO.load(args.model, env=venv)

    # 4) rollout loop
    for ep in range(1, args.episodes + 1):
        obs = venv.reset()
        total_reward = 0.0
        done = False
        terminal_event = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)

            reward = float(unwrap(reward))
            done = bool(unwrap(done))
            info = unwrap(info) or {}

            total_reward += reward
            if "event" in info:
                terminal_event = info["event"]

            venv.render()
            time.sleep(args.delay)

        print(f"Episode {ep:02d}: reward = {total_reward:6.1f}   event = {terminal_event}")

    venv.close()


if __name__ == "__main__":
    main()
