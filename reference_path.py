# -*- coding: utf-8 -*-

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import minimize, root_scalar
from param_fcn import StraightLine, LemniscateCurve, LaneChange, RandPath


""" Examples of usage: 
rpath = ReferencePath("straight", 2.0)
rpath = ReferencePath("eight", 1.0)
rpath = ReferencePath("change", 0.2, 0.3, 0.4)
rpath = ReferencePath("random", {'low':2, 'high':5}, {'low':1.0, 'high':2.0})
"""
class ReferencePath:
    def __init__(self, *args):
        path = args[0]
        if len(args) > 1:
            var = args[1:]
        else:
            var = None
        
        if path.lower() in ["straight", "eight", "change", "random"]:
            if path.lower() == "straight":
                self.path = StraightLine(var)
            elif path.lower() == "eight":
                self.path = LemniscateCurve(var)
            elif path.lower() == "change":
                self.path = LaneChange(var)
            elif path.lower() == "random": 
                self.path = RandPath(var)
                self.path.seed()
                self.path.gen_fcn() # generate a random path
        else:
            print("No such path!")
            
        self.flag_path = path.lower()
        self.len = self.path.len
        
    def X(self, s):
        return self.path.X(s)
    
    def dX(self, s):
        return self.path.dX(s)
    
    def ddX(self, s):
        return self.path.ddX(s)
    
    def Y(self, s):
        return self.path.Y(s)
        
    def dY(self, s):
        return self.path.dY(s)
    
    def ddY(self, s):
        return self.path.ddY(s)
         
    def calc_yaw(self, s):     
        dx, dy = self.dX(s), self.dY(s)   
        return np.arctan2(dy, dx)
    
    def calc_curvature(self, s):
        dx, dy = self.dX(s), self.dY(s)
        ddx, ddy = self.ddX(s), self.ddY(s)  
        
        return (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))

    def find_nearest_point(self, s0, x, y):
        """
            Find the value of s = argmin||(xr, yr) - (x, y)||  
        """
        def distance_squared(_s, *args):
            _x, _y = self.X(_s), self.Y(_s) 
            return (_x - args[0]) ** 2 + (_y - args[1]) ** 2
        
        def derivative(_s, *args):
            _x, _y = self.X(_s), self.Y(_s) 
            _dx, _dy = self.dX(_s), self.dY(_s)  
            return 2 * _dx * (_x - args[0]) + 2 *_dy *(_y-args[1])
    
        result = minimize(distance_squared, x0=s0, jac=derivative, 
                          args=(x, y), method='CG')
        return result.x[0]
    
    def find_lookahead_point(self, s0, x, y, L):
        """
            Find the value of s that satisfies ||(xr, yr) - (x, y)|| = L 
        """
        def distance_squared(_s):  
            return  (self.X(_s) - x)**2 + (self.Y(_s) - y)**2 - L**2
        
        def derivative(_s):
            return 2 * (self.X(_s) - x) * self.dX(_s) + 2 * (self.Y(_s) - y) * self.dY(_s)
        
        sol = root_scalar(distance_squared, x0=s0, fprime=derivative, 
                          method='newton')        
        return sol.root
    
    def calc_lookahead_s0(self, L, N=50):
        # Define the range of s values
        s_values = np.arange(0, self.path.len, 0.01)
        s_values = s_values[:N]
        
        # Compute the arc length corresponding to each s value
        arc_length = np.array([arc_len(s, self.path.dX, self.path.dY) for s in s_values])
        
        # Compute the index of nearest value
        nearest_index = np.argmin(np.abs(arc_length - L))
        return s_values[nearest_index]
        
    def calc_error(self, x, y, yaw, s):
        """
            compute the CTE and orientation error from a reference point
        """
        x_ref, y_ref = self.X(s), self.Y(s) 
        tx, ty = self.dX(s), self.dY(s) 
        
        # cross track error (left to path, e>0; right to path, e<0)
        dx, dy = x - x_ref, y - y_ref
        cte = (dy * tx - dx * ty) / np.sqrt(tx ** 2 + ty ** 2)
        
        # orientation error
        yaw_ref = self.calc_yaw(s)   
        psi = angle_normalize(yaw - yaw_ref)
        return cte, psi
    
    def seed(self, seed=None):
        if self.flag_path == "random":
            self.path.seed(seed)
        else:
            pass
    
    def gen_rand_path(self):
        if self.flag_path == "random":
            x, y = self.path.gen_fcn()
            self.len = self.path.len
            return x, y
        else:
            pass
        
        

def angle_normalize(theta): # in the range *[-pi, pi]*
    return np.arctan2(np.sin(theta), np.cos(theta)) 


                                 
def arc_len(s, dX, dY):
    integrand = lambda u: np.sqrt(dX(u)**2 + dY(u)**2)
    result, _ = quad(integrand, 0, s)
    return result



def arc_parameterized(path):
   # Define the range of s values
   s_values = np.linspace(0, path.len, 200)
   
   # Compute the arc length corresponding to each s value
   arc_length = np.array([arc_len(s, path.dX, path.dY) for s in s_values])
   length = arc_length[-1]
   
   # Interpolated function for x and y with respect to arc length
   x, y = path.X(s_values), path.Y(s_values)
   x_interp = CubicSpline(arc_length, x, bc_type=((2.0, 0.0), (2.0, 0.0))) 
   y_interp = CubicSpline(arc_length, y, bc_type=((2.0, 0.0), (2.0, 0.0)))  
   return x_interp, y_interp, length



class ArcReferencePath: 
    def __init__(self, *args):
        path = args[0]
        if len(args) > 1:
            var = args[1:]
        else:
            var = None
        
        if path.lower() in ["straight", "eight", "change", "random"]:
            if path.lower() == "straight":
                self.path = StraightLine(var)
            elif path.lower() == "eight":
                self.path = LemniscateCurve(var)
            elif path.lower() == "change":
                self.path = LaneChange(var)
            elif path.lower() == "random": 
                self.path = RandPath(var)
                self.path.seed()
                self.path.gen_fcn() # generate a random path
        else:
            print("No such path!")
            
        self.flag_path = path.lower()

        self.X, self.Y, self.len = arc_parameterized(self.path)
        self.dX = self.X.derivative(1)
        self.ddX = self.X.derivative(2)
 
        self.dY = self.Y.derivative(1)
        self.ddY = self.Y.derivative(2)
    
    def calc_yaw(self, s):     
        dx, dy = self.dX(s), self.dY(s)   
        return np.arctan2(dy, dx)
    
    def calc_curvature(self, s):
        dx, dy = self.dX(s), self.dY(s)
        ddx, ddy = self.ddX(s), self.ddY(s)  
        
        return (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
    
    def find_nearest_point(self, s0, x, y):
        """
            Find the value of s = argmin||(xr, yr) - (x, y)||  
        """
        def distance_squared(_s, *args):
            _x, _y = self.X(_s), self.Y(_s) 
            return (_x - args[0]) ** 2 + (_y - args[1]) ** 2
        
        def derivative(_s, *args):
            _x, _y = self.X(_s), self.Y(_s) 
            _dx, _dy = self.dX(_s), self.dY(_s)  
            return 2 * _dx * (_x - args[0]) + 2 *_dy *(_y-args[1])
    
        result = minimize(distance_squared, x0=s0, jac=derivative, 
                          args=(x, y), method='CG')
        return result.x[0]
    
    def find_lookahead_point(self, s0, x, y, L):
        """
            Find the value of s that satisfies ||(xr, yr) - (x, y)|| = L 
        """
        def distance_squared(_s):  
            return  (self.X(_s) - x)**2 + (self.Y(_s) - y)**2 - L**2
        
        def derivative(_s):
            return 2 * (self.X(_s) - x) * self.dX(_s) + 2 * (self.Y(_s) - y) * self.dY(_s)
        
        sol = root_scalar(distance_squared, x0=s0, fprime=derivative, 
                          method='newton')        
        return sol.root
        
    def calc_error(self, x, y, yaw, s):
        """
            compute the CTE and orientation error from a reference point
        """
        x_ref, y_ref = self.X(s), self.Y(s) 
        tx, ty = self.dX(s), self.dY(s) 
        
        # cross track error (left to path, e>0; right to path, e<0)
        dx, dy = x - x_ref, y - y_ref
        cte = (dy * tx - dx * ty) / np.sqrt(tx ** 2 + ty ** 2)
        
        # orientation error
        yaw_ref = self.calc_yaw(s)   
        psi = angle_normalize(yaw - yaw_ref)
        return cte, psi
    
    def seed(self, seed=None):
        if self.flag_path == "random":
            self.path.seed(seed)
        else:
            pass
    
    def gen_rand_path(self):
        if self.flag_path == "random":
            x, y = self.path.gen_fcn()
            self.X, self.Y, self.len = arc_parameterized(self.path)
            self.dX = self.X.derivative(1)
            self.ddX = self.X.derivative(2)
     
            self.dY = self.Y.derivative(1)
            self.ddY = self.Y.derivative(2)
            return x, y
        else:
            pass
        


def calc_distance(x, y, tx, ty):
    dx = x - tx
    dy = y - ty
    return np.hypot(dx, dy)