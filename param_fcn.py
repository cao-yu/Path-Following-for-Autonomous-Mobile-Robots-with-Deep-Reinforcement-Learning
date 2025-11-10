# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import CubicSpline


class StraightLine:
    def __init__(self, *args):
        if args[0] is None:
            self.a = 2.5 # default
        else:
            self.a = args[0][0]
            
        self.len = self.a
        
    def X(self, s):
        return s
    
    def dX(self, s):
        return np.ones_like(s)
    
    def ddX(self, s):
        return np.zeros_like(s)
    
    def Y(self, s):
        return np.zeros_like(s)
        
    def dY(self, s):
        return np.zeros_like(s)
    
    def ddY(self, s):
        return np.zeros_like(s)
    
    
    
class LemniscateCurve:
    def __init__(self, *args):
        if args[0] is None:
            self.a = 1.0 # default
        else:
            self.a = args[0][0]
            
        self.len = 2 * np.pi
        
    def X(self, s):
        return self.a * np.sin(s)
    
    def dX(self, s):
        return self.a * np.cos(s)
    
    def ddX(self, s):
        return - self.a * np.sin(s)
    
    def Y(self, s):
        return self.a * np.sin(s) * np.cos(s)
        
    def dY(self, s):
        return self.a * (np.cos(s) ** 2 - np.sin(s) ** 2)
    
    def ddY(self, s):
        return - self.a * 4 * np.sin(s) * np.cos(s)
    


class LaneChange:
    def __init__(self, *args):
        if args[0] is None:
            self.b = 1.5
            self.c = 1.5
            self.k = 30.0
        else:
            self.b = args[0][0]
            self.c = args[0][1]
            self.k = args[0][2]
            
        self.len = 2 * self.c 
    
    def X(self, s):
        return s
    
    def dX(self, s):
        return np.ones_like(s)
    
    def ddX(self, s):
        return np.zeros_like(s)
    
    def Y(self, s):
        return (self.b) / (1 + np.exp(-self.k*(s-self.c)))
    
    def dY(self, s):
        return (self.b * self.k * np.exp(-self.k*(s-self.c))) / (1 + np.exp(-self.k*(s-self.c))) ** 2
    
    def ddY(self, s):
        numerator = self.b * self.k**2 * np.exp(-self.k * (s - self.c)) * (self.k * (s - self.c) - 1)
        denominator = (1 + np.exp(-self.k * (s - self.c))) ** 3
        return numerator / denominator
        
    
    
    
class RandPath:
    def __init__(self, *args):
        if args[0] is None:               
            self.N = {'low':4, 'high':4}
            self.L = {'low':0.5, 'high':2.0}
        else:
            self.N = args[0][0]
            self.L = args[0][1]
        
        self.seed()
        self.straight_path = StraightLine(None)
        self.ep_cnt = 0
        
    def gen_fcn(self):
        if self.ep_cnt % 10 == 0: # used in training phase
        #if self.ep_cnt % 10 < -1: # for passing this in eval phase
            self.len = self.straight_path.len
            self.X, self.Y = self.straight_path.X, self.straight_path.Y
            self.dX, self.dY = self.straight_path.dX, self.straight_path.dY
            self.ddX, self.ddY = self.straight_path.ddX, self.straight_path.ddY
            x, y = np.array([0, self.len]), np.array([0, 0])
        else:
            while True:
                x, y = self.gen_wpts()
                s = np.linspace(0, self.len, len(x)) 
                
                self.X = CubicSpline(s, x, bc_type=((2.0, 0.0), (2.0, 0.0)))
                self.Y = CubicSpline(s, y, bc_type=((2.0, 0.0), (2.0, 0.0)))
                
                self.dX = self.X.derivative(1)
                self.ddX = self.X.derivative(2)
             
                self.dY = self.Y.derivative(1)
                self.ddY = self.Y.derivative(2)
                
                # check the largest curvature
                s_values = np.linspace(0, s[-1], 200)
                dx, dy = self.dX(s_values), self.dY(s_values)
                ddx, ddy = self.ddX(s_values), self.ddY(s_values)
                k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))       
                if all(abs(num) <= 50.0 for num in k): # in training
                        break
        self.ep_cnt += 1 # put here because it is called when initialized        
        return x, y
    
    def gen_wpts(self):
        x, y = np.zeros(1), np.zeros(1) # Let (x0, y0) = (0, 0)
        Nw = self.np_random.randint(low=self.N['low'], high=self.N['high']+1) # in meters
        while len(x) < Nw + 1:  
            Lw = self.np_random.uniform(low=self.L['low'], high=self.L['high']) # in meters
            theta = 2 * np.pi * self.np_random.rand() # [0, 2pi)
            nx, ny = x[-1] + Lw * np.cos(theta),  y[-1] + Lw * np.sin(theta) # next point     
            
            x = np.append(x, nx) 
            y = np.append(y, ny)  
        self.len = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))[-1]
        return x, y
 
    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        
