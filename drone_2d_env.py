"""
A 2-D Reinforcement Learning environment: a blue drone (circle) must reach a
green goal while avoiding red circular obstacles.
"""

import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# ────────────── Reward & Behavior Constants ──────────────
GOAL_REWARD        = 400.0    # success
COLLISION_PENALTY  = -200.0   # crash
SLOW_PENALTY       = -120.0   # slow‐progress window
STUCK_PENALTY      = -60.0    # early‐stuck
STEP_COST          = -0.5     # per‐step penalty
NO_MOVE_PENALTY    = -1.0     # if action causes no movement

# Dense shaping scales
SHAPING_SCALE_FAR  = 4.0
SHAPING_SCALE_NEAR = 2.0
FAR_THRESHOLD      = 40.0     # px

# Ring breadcrumb bonus
RING_BONUS         = 1.0
RING_WIDTH         = 3.0      # px per ring

# Anti‐oscillation (flip) penalty
FLIP_PENALTY       = -15.0
FLIP_WINDOW        = 6        # last N steps to track
FLIP_MAX           = 3        # flips allowed in window

# Pixel‐loop penalty
LOOP_PENALTY       = -5.0
LOOP_WINDOW        = 20       # last N positions
LOOP_MAX           = 8        # repeats at same pos

# Slow‐progress detection
SLOW_WINDOW        = 20       # last N distances
SLOW_THRESHOLD     = 4.0      # px

# Early‐stuck detection
STUCK_WINDOW       = 15       # frames
STUCK_THRESHOLD    = 0.5      # px

# Goal‐completion margin
SUCCESS_MARGIN     = 5.0      # extra px to count as success

# Wall‐repulsion shaping
WALL_COEFF         = -0.05    # divided by distance‐to‐wall


class Drone2DEnv(gym.Env):
    """
    Gymnasium‐compatible 2D drone navigation environment.
    Action space: Discrete(5) → [stay, up, down, left, right]
    Observation: [drone_x, drone_y, goal_x, goal_y, obs1_x, obs1_y, obs1_r, ...]
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        width=800,
        height=600,
        obstacle_count=5,
        obstacle_radius=20,
        drone_radius=8,
        goal_radius=12,
        step_size=5,
        max_steps=300,
        render_mode=None,
    ):
        super().__init__()

        # ── world geometry ───────────────────────────────
        self.w, self.h = width, height
        self.n_obs, self.max_r = obstacle_count, obstacle_radius
        self.dr_r, self.goal_r = drone_radius, goal_radius
        self.step_size, self.max_steps = step_size, max_steps
        self.render_mode = render_mode

        # ── behavioral windows & thresholds ───────────────
        self.loop_k = LOOP_WINDOW
        self.loop_max = LOOP_MAX
        self.dir_hist_len = FLIP_WINDOW
        self.dir_change_max = FLIP_MAX
        self.no_prog_max = STUCK_WINDOW
        self.prog_window = SLOW_WINDOW
        self.prog_thresh = SLOW_THRESHOLD
        self.margin = SUCCESS_MARGIN

        # ── action & observation spaces ───────────────────
        self.action_space = spaces.Discrete(5)  # 0 stay,1 up,2 down,3 left,4 right
        low  = [0, 0, 0, 0] + [0, 0, 1] * self.n_obs
        high = [self.w, self.h, self.w, self.h] + [self.w, self.h, self.max_r] * self.n_obs
        self.observation_space = spaces.Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

        # ── runtime state ─────────────────────────────────
        self.drone = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.obs = np.zeros((self.n_obs, 3), dtype=np.float32)  # x, y, r
        self.steps = 0

        # history buffers
        self.hist: list[tuple[int,int]] = []   # pixel loop
        self.dir_hist: list[tuple[int,int]] = []
        self.dist_hist: list[float] = []
        self.no_prog = 0
        self.last_action = None

        # ── action → delta mapping ─────────────────────────
        s = self.step_size
        self.delta = {
            0: np.array([ 0,  0], dtype=np.float32),
            1: np.array([ 0, -s], dtype=np.float32),
            2: np.array([ 0,  s], dtype=np.float32),
            3: np.array([-s,  0], dtype=np.float32),
            4: np.array([ s,  0], dtype=np.float32),
        }
        self.opposite = {1: 2, 2: 1, 3: 4, 4: 3}
        self.dir_vec = {k: (int(v[0]!=0), int(v[1]!=0)) for k, v in self.delta.items()}

        # ── rendering colours ──────────────────────────────
        self.c_bg, self.c_goal, self.c_obs, self.c_dr = (
            (255, 255, 255),
            (0, 255,   0),
            (255,  0,   0),
            (0,    0, 255),
        )

        # ── PyGame init (lazy) ────────────────────────────
        self.screen = self.clock = None
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Drone2DEnv")
            self.clock = pygame.time.Clock()

    # ───────────────────────────── helpers ─────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """Flattened observation: [drone, goal, all obstacles]."""
        return np.concatenate(([*self.drone, *self.goal], self.obs.flatten())).astype(np.float32)

    def _place_clear(self, rng: np.random.Generator, r: float) -> np.ndarray:
        """
        Return a random (x, y) at least r away from every obstacle
        (to avoid collisions at spawn).
        """
        while True:
            x = rng.uniform(r,   self.w - r)
            y = rng.uniform(r,   self.h - r)
            # check distance to existing obstacles
            if np.all(np.hypot(x - self.obs[:,0], y - self.obs[:,1]) > (self.obs[:,2] + r)):
                return np.array([x, y], dtype=np.float32)

    # ───────────────────────────── reset ──────────────────────────────
    def reset(self, seed=None, options=None):
        """Reset world: randomize obstacles, drone start, and goal."""
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # clear trackers
        self.steps = 0
        self.hist.clear()
        self.dir_hist.clear()
        self.dist_hist.clear()
        self.no_prog = 0
        self.last_action = None

        # place obstacles
        for i in range(self.n_obs):
            self.obs[i,:2] = self._place_clear(rng, self.max_r)
            self.obs[i, 2] = rng.uniform(4, self.max_r)

        # place drone start
        self.drone = self._place_clear(rng, self.dr_r)

        # place goal, ensure minimum separation
        while True:
            g = self._place_clear(rng, self.goal_r)
            if math.hypot(*(g - self.drone)) > self.goal_r + self.dr_r + 20:
                self.goal = g
                break

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    # ───────────────────────────── step ───────────────────────────────
    def step(self, action: int):
        """
        Apply action, compute reward, and determine done/truncation.
        Returns: obs, reward, terminated, truncated, info
        """
        action = int(action)
        prev = self.drone.copy()
        self.steps += 1
        reward = 0.0

        # ── A. anti‐oscillation (direction‐flip) penalty ───────────
        self.dir_hist.append(self.dir_vec[action])
        if len(self.dir_hist) > self.dir_hist_len:
            self.dir_hist.pop(0)
        flips = sum(
            (dx1 != 0 and dx1 == -dx0) or (dy1 != 0 and dy1 == -dy0)
            for (dx0,dy0), (dx1,dy1) in zip(self.dir_hist, self.dir_hist[1:])
        )
        if flips >= self.dir_change_max:
            reward += FLIP_PENALTY

        # immediate reverse‐action penalty
        if self.last_action is not None and self.opposite.get(self.last_action) == action:
            reward += NO_MOVE_PENALTY
        self.last_action = action

        # ── B. move drone & clamp to bounds ─────────────────────────
        self.drone += self.delta[action]
        self.drone[0] = np.clip(self.drone[0], self.dr_r, self.w - self.dr_r)
        self.drone[1] = np.clip(self.drone[1], self.dr_r, self.h - self.dr_r)
        moved = not np.allclose(prev, self.drone)

        # ── C. dense distance shaping + step cost ──────────────────
        old_d = math.hypot(*(prev   - self.goal))
        new_d = math.hypot(*(self.drone - self.goal))
        scale = SHAPING_SCALE_FAR if new_d > FAR_THRESHOLD else SHAPING_SCALE_NEAR
        reward += (old_d - new_d) * scale + STEP_COST
        if not moved:
            reward += NO_MOVE_PENALTY

        # ── D. ring breadcrumb bonus ───────────────────────────────
        if int(old_d // RING_WIDTH) > int(new_d // RING_WIDTH):
            reward += RING_BONUS

        terminated = truncated = False
        info = {}

        # ── E. pixel-loop penalty ─────────────────────────────────
        self.hist.append(tuple(self.drone.astype(int)))
        if len(self.hist) > LOOP_WINDOW:
            self.hist.pop(0)
        if self.hist.count(self.hist[-1]) >= LOOP_MAX:
            reward += LOOP_PENALTY

        # ── F. early‐stuck truncation ─────────────────────────────
        if old_d - new_d > STUCK_THRESHOLD:
            self.no_prog = 0
        else:
            self.no_prog += 1
        if self.no_prog >= STUCK_WINDOW:
            reward = STUCK_PENALTY
            truncated = True
            info["event"] = "stuck"

        # ── G. slow‐progress window truncation ────────────────────
        self.dist_hist.append(new_d)
        if len(self.dist_hist) > SLOW_WINDOW:
            self.dist_hist.pop(0)
        if len(self.dist_hist) == SLOW_WINDOW and not (terminated or truncated):
            if self.dist_hist[0] - self.dist_hist[-1] < SLOW_THRESHOLD:
                reward = SLOW_PENALTY
                truncated = True
                info["event"] = "slow"

        # ── H. collision check ─────────────────────────────────────
        for ox, oy, orad in self.obs:
            if math.hypot(self.drone[0]-ox, self.drone[1]-oy) <= self.dr_r + orad:
                reward = COLLISION_PENALTY
                terminated = True
                info["event"] = "collision"
                break

        # ── I. success halo check ─────────────────────────────────
        if new_d <= self.dr_r + self.goal_r + self.margin:
            reward += RING_BONUS  # small per-frame bonus in halo
            if not terminated:
                reward = GOAL_REWARD
                terminated = True
                info["event"] = "goal"

        # ── J. global timeout ─────────────────────────────────────
        if not (terminated or truncated) and self.steps >= self.max_steps:
            truncated = True
            info["event"] = "timeout"

        # ── K. wall‐repulsion shaping ──────────────────────────────
        dw = min(
            self.drone[0] - self.dr_r,
            self.w - self.dr_r - self.drone[0],
            self.drone[1] - self.dr_r,
            self.h - self.dr_r - self.drone[1],
        )
        reward += WALL_COEFF / (dw + 1e-3)

        # ── render if requested ───────────────────────────────────
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    # ───────────────────────── render / close ─────────────────────────–
    def render(self):
        """Display a frame in a PyGame window."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.w, self.h))
            self.clock = pygame.time.Clock()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return

        self.screen.fill(self.c_bg)
        # draw goal, obstacles, and drone
        pygame.draw.circle(self.screen, self.c_goal, self.goal.astype(int), self.goal_r)
        for ox, oy, orad in self.obs:
            pygame.draw.circle(self.screen, self.c_obs, (int(ox), int(oy)), int(orad))
        pygame.draw.circle(self.screen, self.c_dr, self.drone.astype(int), self.dr_r)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Close the PyGame window, if open."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
