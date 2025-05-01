# eval.py
import os, time, argparse, numpy as np
from stable_baselines3           import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_2d_env import Drone2DEnv

def unwrap(x):
    return x[0] if isinstance(x, (np.ndarray, list, tuple)) else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--episodes", type=int,   default=25,   help="episodes")
    ap.add_argument("-d", "--delay",    type=float, default=0.02, help="sec / frame")
    args = ap.parse_args()

    venv = DummyVecEnv([lambda: Drone2DEnv(render_mode="human")])
    venv = VecNormalize.load("train_logs/vec_normalize.pkl", venv)
    venv.training    = False
    venv.norm_reward = False

    model_path = "train_logs/ppo_drone_model.zip"
    model = PPO.load(model_path, env=venv)

    for ep in range(1, args.episodes+1):
        obs = venv.reset()
        total = 0.0; done=False; event=None
        while not done:
            act,_ = model.predict(obs, deterministic=True)
            obs, rew, done, info = venv.step(act)
            rew   = float(unwrap(rew)); done=bool(unwrap(done)); info=unwrap(info)
            total += rew
            if info and "event" in info: event = info["event"]
            venv.render(); time.sleep(args.delay)
        print(f"Ep {ep}: reward={total:.1f}  event={event}")
    venv.close()


if __name__ == "__main__":
    main()
