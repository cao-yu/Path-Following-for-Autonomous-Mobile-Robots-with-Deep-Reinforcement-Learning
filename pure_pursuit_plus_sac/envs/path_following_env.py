# -*- coding: utf-8 -*-

import sys
from pathlib import Path
GRANDPARENT = Path(__file__).resolve().parents[2] 
gp_str = str(GRANDPARENT)
if gp_str not in sys.path:
    sys.path.insert(0, gp_str)

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import gymnasium as gym
from gymnasium import spaces

from kinematics import Unicycle, Logger
from reference_path import ArcReferencePath


# Action 
MAX_V_DOT_ACC = 0.3 # [m/s^2]
MAX_V_DOT_DEACC = -0.5 # [m/s^2] 

# pure pursuit parameters
DEFAULT_L = 0.2 # [m] lookahead distance
DEFAULT_K = 0.0 # proportional gain

"""
Env. for training an adaptive velocity controller
with the presence of pure pursuit steering controller
"""


class PathFollowingEnv(gym.Env): 
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, path="random"):
        super().__init__()
        
        # import model & data logger
        self.mdl = Unicycle()
        self.logger = Logger()
        
        # action space
        self.action_space = spaces.Box(low=MAX_V_DOT_DEACC, high=MAX_V_DOT_ACC, 
                                shape=(1,), dtype=np.float32)
        
        # observation space (cte, psi, mdl.v, mdl.omega, psi1)  
        high = np.array([np.inf, np.pi, self.mdl.max_v, self.mdl.max_omega, np.pi], 
                        dtype=np.float32)
        low = np.array([-np.inf, -np.pi, 0.0, -self.mdl.max_omega, -np.pi], 
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.float32)
        
        # reference path
        self.path = RefPath(path)
        
        # random seed
        self.seed()
        
        # initialize step_cnt as an instance variable
        self.step_cnt = 0
        
        # my render
        self.ax = None
        self.cbar = None
       
    def step(self, action): 
        self.step_cnt += 1 # Increment instance variable
        self.logger.append(self.mdl) 
        
        v_ref = self.mdl.v + action[0] * self.mdl.dt
        omega_ref = pure_pursuit_control(self.mdl, self.path.path, \
                                         self.path.sl)     
        
        if self.mdl.tau > 0.0 or self.mdl.k > 0.0:
            self.mdl.update_tau(v_ref, omega_ref) 
        else:
            self.mdl.update(v_ref, omega_ref) # used in training
        
        obs = self._get_obs()
        reward = self.reward(obs)
        #terminated = self.check_termination(obs) # default: False in training phase
        truncated = self.check_truncation()
        return obs, reward, False, truncated, {} # state, reward, done, info
    
    def reward(self, obs): 
        rew = - 5.0 * np.abs(obs[0]) \
            + 2.5 * obs[2] * (1.0 - 5.0 * np.abs(obs[0]))
        return rew - 0.2 * float(obs[2] < 1e-6)
    
    def check_truncation(self):
        done = self.path.check_truncation() \
            or self.step_cnt >= 400
        return done
    
    def check_termination(self, obs):
        if np.abs(obs[0]) >= 0.2 or np.cos(obs[1]) <= 0.0: 
            return True 
        else:
            return False
    
    def reset(self, T=0.0, k=0, seed=None, options=None):
        super().reset(seed=seed) 
        
        self.step_cnt = 0 # Reset instance variable  
        self.path.reset()  
        self.mdl.reset(yaw=self.path.path.calc_yaw(0.0), T=T, k=int(k), flag="random")     
        self.logger.reset()
        self.ax = None
        return self._get_obs(), {}
    
    def seed(self, seed=None):
        self.mdl.seed(seed=seed)
        self.path.seed(seed=seed)
    
    def _get_obs(self):     
        return self.path._get_obs(self.mdl)
    
    def render(self, w=0.172, h=0.2, wh=0.15, ww=0.03, alpha=0.5): # customized animation
        if self.ax is None:
           # initialization
           plt.clf()
           plt.ion()
           self.ax = plt.gca()
                  
        # robot
        xy = [self.mdl.x - 0.5*h, self.mdl.y - 0.5*w] # body
        body = patches.Rectangle(xy, h, w, angle=np.rad2deg(self.mdl.yaw), 
                               rotation_point=(self.mdl.x, self.mdl.y), 
                               fc='b', ec='k', alpha=alpha) 
        
        xy = [self.mdl.x - 0.5*wh, self.mdl.y + 0.5*w + 0.01] # left wheel
        lw = patches.Rectangle(xy, wh, ww, angle=np.rad2deg(self.mdl.yaw), 
                             rotation_point=(self.mdl.x, self.mdl.y), 
                             fc='gray', ec='k', alpha=alpha) 
      
        xy = [self.mdl.x - 0.5*wh, self.mdl.y - 0.5*w - ww - 0.01] # right wheel
        rw = patches.Rectangle(xy, wh, ww, angle=np.rad2deg(self.mdl.yaw), 
                            rotation_point=(self.mdl.x, self.mdl.y), 
                            fc='gray', ec='k', alpha=alpha) 
              
        self.ax.cla() # clear    
        self.ax.set_xlim(min(self.path.x)-0.2, max(self.path.x)+0.2)
        self.ax.set_ylim(min(self.path.y)-0.2, max(self.path.y)+0.2)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_title('Path Following')
        self.ax.set_aspect('equal')
        self.ax.grid()
        plt.title(f"v:{self.mdl.v:.2f} m/s, "
                  r"$\omega$" f":{self.mdl.omega:.2f} rad/s")
        
        self.ax.plot(self.path.x, self.path.y, 'k-', linewidth=2.5) # ref path
        #self.ax.plot(self.logger.x, self.logger.y, '--', color='lime', linewidth=2.5) # trajectory    

        # ===== trajectory with velocity color =====
        sac_x = self.logger.x
        sac_y = self.logger.y
        sac_v = self.logger.v

        sc = self.ax.scatter(
            sac_x, sac_y,
            s=60,
            c=sac_v,
            edgecolors='face',
            cmap='jet',
            vmin=0.0,
            vmax=self.mdl.max_v
        )

        # Add colorbar only once
        if self.cbar is None:
            self.cbar = self.ax.figure.colorbar(sc, ax=self.ax)
            self.cbar.ax.tick_params(labelsize=12)
            self.cbar.set_label('Velocity [m/s]', fontsize=12)
        # ==========================================
        
        self.ax.add_patch(body)
        self.ax.add_patch(lw)
        self.ax.add_patch(rw)
        
        # nearest point
        self.ax.plot(self.path.path.X(self.path.sn), 
                     self.path.path.Y(self.path.sn), 'bo',
                     markeredgewidth=3, ms=8) 
        
        # lookahead point  
        self.ax.plot(self.path.path.X(self.path.sl),
                     self.path.path.Y(self.path.sl), 'rx',
                     markeredgewidth=3, ms=8)
        
        self.ax.figure.canvas.draw()
        plt.pause(0.02)



def pure_pursuit_control(mdl, target_path, sl):
    tx, ty = target_path.X(sl), target_path.Y(sl)
    alpha = math.atan2(ty - mdl.y, tx - mdl.x) - mdl.yaw
    omega = 2 * mdl.v * math.sin(alpha) / mdl.calc_distance(tx, ty)
    return omega



class RefPath:
    def __init__(self, path="random"):
        self.path = ArcReferencePath(path)
        self.flag_path = path.lower()
        
        s = np.linspace(0.0, self.path.len, 200)
        self.x, self.y = self.path.X(s), self.path.Y(s)
        
        self.sn = 0.0
        self.sl = DEFAULT_L
        #self.sl2 = 2 * DEFAULT_L
        
        self.seed()
        
    def _get_obs(self, mdl):  
        # nearest point
        self.sn = self.path.find_nearest_point(self.sn, mdl.x, mdl.y)

        # look-ahead(anticipation) point
        L = DEFAULT_L + DEFAULT_K * mdl.v
        sl = self.sn + L
        if self.sl < sl:
            self.sl = sl
        if self.sl > self.path.len:
            self.sl = self.path.len
        
        # current error
        cte, psi = self.path.calc_error(mdl.x, mdl.y, mdl.yaw, self.sn)
        
        # anticipation error
        _, psi1 = self.path.calc_error(mdl.x, mdl.y, mdl.yaw, self.sl)
       
        # concat
        obs = np.array([cte, psi, mdl.v, mdl.omega, psi1], dtype=np.float32)
        return obs

    def check_truncation(self):
        done = self.path.len - self.sn <  0.02
        return done
    
    def seed(self, seed=None):
        self.path.seed(seed=seed)
    
    def reset(self):
        if self.flag_path == "random":
            self.path.gen_rand_path() # new random path

            # for rendering
            s = np.linspace(0.0, self.path.len, 200)
            self.x, self.y = self.path.X(s), self.path.Y(s)
            
        self.sn = 0.0

        self.sl = DEFAULT_L


