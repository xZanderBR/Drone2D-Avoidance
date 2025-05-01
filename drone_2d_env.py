import math, numpy as np, pygame, gymnasium as gym
from gymnasium import spaces

class Drone2DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # ───────────────────────────── init
    def __init__(self,
                 width=800, height=600,
                 obstacle_count=5, obstacle_radius=20,
                 drone_radius=8, goal_radius=12,
                 step_size=5, max_steps=300,
                 loop_k=20, loop_max=8,
                 no_progress_max=15,
                 success_margin=5.0,
                 rev_penalty=True,
                 render_mode=None):

        super().__init__()
        # geometry
        self.w,self.h = width,height
        self.n_obs,self.max_r  = obstacle_count, obstacle_radius
        self.dr_r,self.goal_r  = drone_radius,  goal_radius
        self.step_size,self.max_steps = step_size,max_steps

        # anti-oscillation
        self.loop_k, self.loop_max   = loop_k, loop_max
        self.dir_hist_len            = 6      # 0.2 s window
        self.dir_change_max          = 3
        self.no_prog_max             = no_progress_max
        # NEW slow-progress window
        self.prog_window  = 25       # 0.8 s window
        self.prog_thresh  = 4.0      # must close ≥4 px

        self.margin       = success_margin
        self.rev_penalty  = rev_penalty
        self.render_mode  = render_mode

        # spaces
        self.action_space = spaces.Discrete(5)
        low  = [0,0,0,0] + [0,0,1]*self.n_obs
        high = [self.w,self.h,self.w,self.h] + [self.w,self.h,self.max_r]*self.n_obs
        self.observation_space = spaces.Box(np.array(low ,dtype=np.float32),
                                            np.array(high,dtype=np.float32),
                                            dtype=np.float32)

        # runtime
        self.drone = np.zeros(2,dtype=np.float32)
        self.goal  = np.zeros(2,dtype=np.float32)
        self.obs   = np.zeros((self.n_obs,3),dtype=np.float32)
        self.steps = 0
        self.hist,self.dir_hist,self.dist_hist = [],[],[]
        self.no_prog = 0
        self.last_action = None

        # movement
        s=self.step_size
        self.delta={0:np.array([0,0],dtype=np.float32),
                    1:np.array([0,-s],dtype=np.float32),
                    2:np.array([0, s],dtype=np.float32),
                    3:np.array([-s,0],dtype=np.float32),
                    4:np.array([ s,0],dtype=np.float32)}
        self.opposite={1:2,2:1,3:4,4:3}
        self.dir_vec={0:(0,0),1:(0,-1),2:(0,1),3:(-1,0),4:(1,0)}

        # colours
        self.c_bg,self.c_goal,self.c_obs,self.c_dr=(255,255,255),(0,255,0),(255,0,0),(0,0,255)

        # pygame
        self.screen=self.clock=None
        if render_mode=="human":
            pygame.init()
            self.screen=pygame.display.set_mode((self.w,self.h))
            pygame.display.set_caption("Drone2DEnv"); self.clock=pygame.time.Clock()

    # ─────────────────────────── helpers
    def _get_obs(self):
        return np.concatenate(([*self.drone,*self.goal], self.obs.flatten())).astype(np.float32)

    def _place_clear(self, rng, r):
        while True:
            x=rng.uniform(r,self.w-r); y=rng.uniform(r,self.h-r)
            if np.all(np.hypot(x-self.obs[:,0],y-self.obs[:,1]) > (self.obs[:,2]+r)):
                return np.array([x,y],dtype=np.float32)

    # ─────────────────────────── reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng=np.random.default_rng(seed)
        self.steps=0
        self.hist,self.dir_hist,self.dist_hist=[],[],[]
        self.no_prog=0; self.last_action=None

        for i in range(self.n_obs):
            self.obs[i,:2]=self._place_clear(rng,self.max_r)
            self.obs[i, 2]=rng.uniform(4,self.max_r)

        self.drone=self._place_clear(rng,self.dr_r)
        while True:
            g=self._place_clear(rng,self.goal_r)
            if np.hypot(*(g-self.drone))>self.goal_r+self.dr_r+20:
                self.goal=g; break

        if self.render_mode=="human": self.render()
        return self._get_obs(), {}

    # ─────────────────────────── step
    def step(self, action):
        action=int(action)
        prev=self.drone.copy(); self.steps+=1
        reward=0.0

        # direction-window anti-oscillation
        self.dir_hist.append(self.dir_vec[action])
        if len(self.dir_hist)>self.dir_hist_len: self.dir_hist.pop(0)
        flips=sum((dx1!=0 and dx1==-dx0) or (dy1!=0 and dy1==-dy0)
                  for (dx0,dy0),(dx1,dy1) in zip(self.dir_hist,self.dir_hist[1:]))
        if flips>=self.dir_change_max: reward-=10.0

        # reverse one-step penalty
        if self.rev_penalty and self.last_action is not None and self.opposite.get(self.last_action)==action:
            reward-=1.0
        self.last_action=action

        # move
        self.drone+=self.delta[action]
        self.drone[0]=np.clip(self.drone[0],self.dr_r,self.w-self.dr_r)
        self.drone[1]=np.clip(self.drone[1],self.dr_r,self.h-self.dr_r)
        moved=not np.allclose(prev,self.drone)

        # distance shaping
        old_d=np.hypot(*(prev       - self.goal))
        new_d=np.hypot(*(self.drone - self.goal))
        scale=4.0 if new_d>40 else 2.0
        reward+=(old_d-new_d)*scale - 0.5
        if not moved: reward-=1.0

        # ring bonus every 3 px
        if int(old_d//3.0) > int(new_d//3.0): reward+=1.0

        terminated=truncated=False; info={}

        # loop-penalty on pixel stagnation
        self.hist.append(tuple(self.drone.astype(int)))
        if len(self.hist)>self.loop_k: self.hist.pop(0)
        if self.hist.count(self.hist[-1])>=self.loop_max: reward-=5.0

        # early-stuck truncation
        thresh=0.5
        self.no_prog = 0 if old_d-new_d>thresh else self.no_prog+1
        if self.no_prog>=self.no_prog_max and not terminated:
            reward-=50.0; truncated=True; info["event"]="stuck"

        # NEW  slow-progress window
        self.dist_hist.append(new_d)
        if len(self.dist_hist)>self.prog_window: self.dist_hist.pop(0)
        if len(self.dist_hist)==self.prog_window:
            if self.dist_hist[0]-self.dist_hist[-1] < self.prog_thresh and not terminated:
                reward-=100.0; truncated=True; info["event"]="timeout"

        # collision
        for ox,oy,orad in self.obs:
            if np.hypot(self.drone[0]-ox,self.drone[1]-oy) <= self.dr_r+orad:
                reward=-200.0; terminated=True; info["event"]="collision"; break

        # goal + halo bonus
        if new_d<=self.dr_r+self.goal_r+self.margin:
            reward+=2.0
            if not terminated:
                reward=400.0; terminated=True; info["event"]="goal"

        # timeout
        if not terminated and self.steps>=self.max_steps:
            truncated=True; info["event"]="timeout"

        # wall-repulsion
        dw=min(self.drone[0]-self.dr_r,
               self.w-self.dr_r-self.drone[0],
               self.drone[1]-self.dr_r,
               self.h-self.dr_r-self.drone[1])
        reward-=0.05/(dw+1e-3)

        if self.render_mode=="human": self.render()
        return self._get_obs(), reward, terminated, truncated, info

    # ───────────────────── render / close
    def render(self):
        if self.screen is None:
            pygame.init(); self.screen=pygame.display.set_mode((self.w,self.h)); self.clock=pygame.time.Clock()
        for e in pygame.event.get():
            if e.type==pygame.QUIT: pygame.quit(); self.screen=None; return
        self.screen.fill(self.c_bg)
        pygame.draw.circle(self.screen,self.c_goal,self.goal.astype(int),self.goal_r)
        for ox,oy,orad in self.obs:
            pygame.draw.circle(self.screen,self.c_obs,(int(ox),int(oy)),int(orad))
        pygame.draw.circle(self.screen,self.c_dr,self.drone.astype(int),self.dr_r)
        pygame.display.flip(); self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit(); self.screen=None
