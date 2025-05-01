# Drone2D-Avoidance

A minimalist project where a **blue drone** must reach a **green goal** while dodging **red circular obstacles** in a 2‑D world.  
The environment (`drone_2d_env.py`) is Gymnasium‑compatible and rendered with PyGame; training uses **PPO** from Stable‑Baselines 3.

---

## 1  Install dependencies
```bash
# (optional) create & activate a virtual‑env …

pip install -r requirements.txt
# Gymnasium ‧ PyGame ‧ Stable‑Baselines3 ‧ PyTorch ‧ NumPy
```
*(If you’re on Apple‑silicon or GPU, install the matching PyTorch build first, then run the command above.)*

---

## 2  Train from scratch
```bash
python train.py
```
* Logs stream to the console **and** TensorBoard (`ppo_drone_tb/`).  
* Default hyper‑params  
  `learning_rate=5e-4 · n_steps=1024 · ent_coef=0.01 · total_timesteps=1_000_000`
* View curves in your browser:
  ```bash
  tensorboard --logdir ppo_drone_tb
  ```

The best policy is auto‑saved to `train_logs/best_model.zip` plus its normaliser `vec_normalize.pkl`.

---

## 3  Continue training
```bash
python continue_train.py         # adds another 150 k steps by default
```
The script reloads the latest checkpoint, keeps the global timestep counter, and writes `ppo_drone_model_continued.zip`.

---

## 4  Watch the agent
```bash
python eval.py --episodes 10 --delay 0.05

# or point explicitly at files:
python eval.py \
  --model train_logs/ppo_drone_model_continued.zip \
  --norm  train_logs/vec_normalize_continued.pkl \
  --episodes 20 --delay 0.03
```
* A PyGame window opens; the blue dot moves in real time.  
* Console prints results like  
  `Ep 3: reward = 196.0  event = goal`  
  Possible events: **goal**, **collision**, **timeout**, **stuck**.

---

## 5  Customise the world
```python
from drone_2d_env import Drone2DEnv
env = Drone2DEnv(width=800,
                 height=600,
                 obstacle_count=8,
                 obstacle_radius=20,
                 drone_radius=8,
                 goal_radius=12,
                 max_steps=400)
```

Geometry tweaks work with an existing policy; a short fine‑tune (~20 k–50 k steps) improves performance.

---

## 6  Reward anatomy

| Component | Value |
|-----------|-------|
| Dense shaping | `(old_dist − new_dist) × scale` (scale = 4 when far >40 px, else 2) |
| Ring bonus | +1 every 3 px closer |
| Wall repulsion | −0.05 / distance‑to‑nearest‑wall |
| Flip penalty | −10 if >3 direction flips in last 6 steps |
| Loop penalty | −5 if same pixel occurs ≥8× in 20‑frame window |
| Early‑stuck | −50 & truncate after 15 frames (<0.5 px progress) |
| Slow progress | −20 & truncate if <4 px closer over last 25 frames |
| Goal / Collision | +200  /  −200 |

TensorBoard’s *rollout/ep_rew_mean* is the sum of all components per episode.

---

## 7  External libraries & resources

| Library | Purpose | Key References |
|---------|---------|----------------|
| **Gymnasium** | RL interface (env API) | <https://gymnasium.farama.org> |
| **PyGame** | 2‑D rendering window | <https://www.pygame.org/docs/> |
| **Stable‑Baselines3** | PPO algorithm, evaluation callbacks, VecNormalize | <https://stable-baselines3.readthedocs.io> |
| **PyTorch** | Neural‑network backend used by SB3 (MLPPolicy) | <https://pytorch.org> |
| **NumPy** | Vector math, RNG utilities | <https://numpy.org> |

---

All are installed automatically via `pip install -r requirements.txt`.
Enjoy exploring!
