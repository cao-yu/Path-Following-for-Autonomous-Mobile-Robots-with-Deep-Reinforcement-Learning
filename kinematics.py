# -*- coding: utf-8 -*-

import math
import numpy as np

from collections import deque
from scipy.signal import cont2discrete


# default paramters of the mobile robot
DEFAULT_WB = 0.172 # wheel base [m]
DEFAULT_V_MAX = 0.4 # maximum linear velocity (without turning) [m/s]
#DEFAULT_OMEGA_MAX = 2 * DEFAULT_V_MAX / DEFAULT_WB  # theoretical maximum rotational velocity [rad/s]
DEFAULT_OMEGA_MAX = 1.0

# random range of starting posuture
DEFAULT_X = 0.1 # [m]
DEFAULT_Y = 0.1 # [m]
DEFAULT_TH = 5.0 # [deg]


"Kinematics Model of a differentially steered mobile robot"
"""
% -------------------------------------------------------
% Positive directions
% the world coordinate(static):
%    x: right [m]
%    y: up [m]
%    theta: counter-clockwise [rad]
%    
% the robot coordinate(inertial):
%    x: forward [m]
%    y: left [m]
%    theta: counter-clockwise [rad]
% -------------------------------------------------------     
"""

"Maximum Velocities"
"""
% -------------------------------------------------------
% The velocity of the outer wheel should not exceed 
% the maximum wheel velocity:
%    v + omega * wb/2 < v_max
% Assume that v_max = 0.5 m/s
% we can choose (v, omega) = (0.4, 1.0)
% -------------------------------------------------------     
"""

class Unicycle:  
    def __init__(self, dt=0.05, WB=DEFAULT_WB,
                 max_v=DEFAULT_V_MAX, max_omega=DEFAULT_OMEGA_MAX,
                 x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0,
                 T=0.0, k=0):
        
        # initialization
        self.dt = dt # sampling period [s]
        self.WB = WB # wheel base [m]
        self.max_v = max_v # maximum linear velocity [m/s]
        self.max_omega = max_omega # maximum rotational velocity [rad/s]

        # initial conditions
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.omega = omega
        self.v_ref = 0.0
        self.omega_ref = 0.0
           
        # state transition-related
        self.tau = T  # time constant
        self.k = k # dead time (index shift)
        if self.tau > 0.0 or self.k > 0.0:
            num = [1]
            den = [T, 1]

            # discrete transfer function
            tfd = cont2discrete((num, den), self.dt, method='bilinear')
            temp = tfd[0][0]
            first_nonzero_index = next((i for i, x in enumerate(temp) if x != 0), None)

            if first_nonzero_index is not None:
                self.num = temp[first_nonzero_index:]
            else:
                print("The array is all zeros, and the new array is empty!")
            
            self.den = tfd[1]   
            
            # (right in, left out) 
            self.vr_prev = deque([0.0] * (len(self.den)-1), 
                                maxlen = (len(self.den)-1)) # the latest not included
            self.vl_prev = deque([0.0] * (len(self.den)-1),
                                    maxlen = (len(self.den)-1)) # the latest not included
            self.vr_ref = deque([0.0] * (k+len(self.num)), 
                               maxlen = (k+len(self.num))) 
            self.vl_ref = deque([0.0] * (k+len(self.num)),
                                   maxlen = (k+len(self.num))) 
        
        # random number generator
        self.seed()
    
    def update(self, v_ref, omega_ref): 
        # range clipping
        self.v = np.clip(v_ref, 0.0, self.max_v)
        self.omega = np.clip(omega_ref, -self.max_omega, self.max_omega)
        
        # state transition
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt    
        self.yaw += self.omega * self.dt
        self.yaw = angle_normalize(self.yaw)
        
    def update_tau(self, v_ref, omega_ref): 
        if self.tau == 0.0:
            print("Time constant is not set")
        if self.k == 0.0:
            print("Delay is not set")
            
        # range clipping
        v_ref = np.clip(v_ref, 0.0, self.max_v)
        omega_ref = np.clip(omega_ref, -self.max_omega, self.max_omega)
        self.v_ref = v_ref
        self.omega_ref = omega_ref 
   
        # tranform to wheel velocities
        vr_ref, vl_ref = self.body_2_wheel(v_ref, omega_ref)
        self.vr_ref.append(vr_ref)
        self.vl_ref.append(vl_ref)
 
        # wheel velocties transition
        vr = self.vel_trans(self.vr_ref, self.vr_prev)
        vl = self.vel_trans(self.vl_ref, self.vl_prev)
        self.vr_prev.append(vr)
        self.vl_prev.append(vl)
        
        # transform to body velocities
        self.v, self.omega = self.wheel_2_body(vr, vl)
    
        # state transition
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt    
        self.yaw += self.omega * self.dt
        self.yaw = angle_normalize(self.yaw)
        
    def body_2_wheel(self, v, omega):
        vr = v + 0.5 * omega * self.WB
        vl = v - 0.5 * omega * self.WB
        
        return vr, vl
        
    def wheel_2_body(self, vr, vl):
        v = 0.5 * (vr + vl)
        omega = (vr - vl) / self.WB
        
        return v, omega
    
    def vel_trans(self, u, y):
        u_terms = np.dot(self.num, 
                         np.array(u)[-self.k-1: 
                                     -self.k-1-len(self.num):
                                         -1])
        y_terms = np.dot(-self.den[1::], 
                         np.array(y)[::-1])
        
        return u_terms + y_terms

    def calc_distance(self, point_x, point_y):        
        dx = self.x - point_x
        dy = self.y - point_y
        
        return math.hypot(dx, dy)
    
    def error_posture(self, x_ref, y_ref, yaw_ref):
        x_e = np.cos(self.yaw) * (x_ref - self.x) + np.sin(self.yaw) * (y_ref - self.y)
        y_e = -np.sin(self.yaw) * (x_ref - self.x) + np.cos(self.yaw) * (y_ref - self.y)
        yaw_e = angle_normalize(yaw_ref - self.yaw)
        
        return x_e, y_e, yaw_e
        
    def reset(self, flag="zero",
              x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0,
              T=0.0, k=0):   
        if flag == "random":
            x += self.np_random.uniform(low = -DEFAULT_X, 
                                   high = DEFAULT_X)
            y += self.np_random.uniform(low = -DEFAULT_Y, 
                                   high = DEFAULT_Y)
            yaw += self.np_random.uniform(low = -np.deg2rad(DEFAULT_TH), 
                                     high = np.deg2rad(DEFAULT_X))

        self.x = x
        self.y = y
        self.yaw = yaw       
        self.v = v
        self.omega = omega
        self.v_ref = 0.0
        self.omega_ref = 0.0
        self.tau = T
        self.k = k
        
        if self.tau > 0.0 or self.k > 0.0:
            num = [1]
            den = [T, 1]

            # discrete transfer function
            tfd = cont2discrete((num, den), self.dt, method='bilinear')
            temp = tfd[0][0]
            first_nonzero_index = next((i for i, x in enumerate(temp) if x != 0), None)

            if first_nonzero_index is not None:
                self.num = temp[first_nonzero_index:]
            else:
                print("The array is all zeros and the new array is empty")
            self.den = tfd[1]  
            
            # state transition-related 
            self.vr_prev = deque([0.0] * (len(self.den)-1), 
                                maxlen = (len(self.den)-1)) # the latest not included
            self.vl_prev = deque([0.0] * (len(self.den)-1),
                                    maxlen = (len(self.den)-1)) # the latest not included
            self.vr_ref = deque([0.0] * (self.k+len(self.num)), 
                               maxlen = (self.k+len(self.num))) 
            self.vl_ref = deque([0.0] * (self.k+len(self.num)),
                                   maxlen = (self.k+len(self.num))) 

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        


def angle_normalize(theta): # in the range *[-pi, pi]*
    return math.atan2(math.sin(theta), math.cos(theta)) 



class Logger:
    def __init__(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.omega = []
        self.v_ref = []
        self.omega_ref = []

    def append(self, mdl):
        # time
        if self.t: # not empty
            self.t.append(self.t[-1] + mdl.dt)
        else:
            self.t = [0.0]
            
        self.x.append(mdl.x)
        self.y.append(mdl.y)
        self.yaw.append(mdl.yaw)
        self.v.append(mdl.v)
        self.omega.append(mdl.omega)
        self.v_ref.append(mdl.v_ref)
        self.omega_ref.append(mdl.omega_ref)
        
    def reset(self):
        self.t = []
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.omega = []
        self.v_ref = []
        self.omega_ref = []