# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patch

from kinematics import Unicycle, Logger
from reference_path import ReferencePath
from control_law import pure_pursuit_control


T = 60.0  # max simulation time [s]
dt = 0.05 # [s]
v_ref = 0.4 # [m/s]


 
"""-------animation-------"""
def update(i):  
    # frame initialization
    plt.cla()
    plt.rcParams["font.size"] = 12
    #ax.set_xlim(state.x[i] - 0.5, state.x[i] + 0.5) # robot centered
    #ax.set_ylim(state.y[i] - 0.5, state.y[i] + 0.5)
    ax.set_xlim(min(rx) - 0.2, max(rx) + 0.2)
    ax.set_ylim(min(ry) - 0.2, max(ry) + 0.2)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True) 
    plt.title(f"t:{state.t[i]:.2f} [s], v:{state.v[i]:.2f} [m/s], "
              r"$\omega$" f":{state.omega[i-1]:.2f} [rad/s]")
    
    # reference path
    plt.plot(rx, ry, 'k', lw=3, label="Reference")
    
    # nearest point
    plt.plot(rpath.X(ls_sn[i]), rpath.Y(ls_sn[i]), 'bo', markeredgewidth=3, ms=10, label="Nearest Point")
    
    # lookahead point
    plt.plot(rpath.X(ls_sl[i]), rpath.Y(ls_sl[i]), 'rx', markeredgewidth=3, ms=10, label="Lookahead Point")
   
    # trajectory
    plt.plot(state.x[0: i+1], state.y[0: i+1], '--', lw=3, color='lime', label="Trajectory")
    
    # robot direction
    #plt.arrow(state.x[i], state.y[i], 0.2*np.cos(state.yaw[i]), 0.2*np.sin(state.yaw[i]),
    #          width=0.005, head_width=0.05, head_length=0.05,
    #          fc='b', ec='k', alpha=0.5)
    
    # robot
    body, leftw, rightw = plot_robot(state.x[i], state.y[i], state.yaw[i],
                                     clr='b',alpha=0.5)
    bd = ax.add_patch(body)
    lw = ax.add_patch(leftw)
    rw = ax.add_patch(rightw)
    
    plt.legend(loc='upper left', ncol=1)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
      
    return bd, lw, rw,
   

def plot_robot(x, y, th, clr, w=0.15, h=0.2, wh=0.15, ww=0.03, alpha=None):
    epsilon = 0.01 # distance between 'body' and 'wheel'
      
    # robot body
    xy = [x - 0.5*h, y - 0.5*w]
    body = patch.Rectangle(xy, h, w, angle=np.rad2deg(th), 
                           rotation_point=(x, y), fc=clr, ec='k', alpha=alpha) 
    
    # left wheel
    xy = [x - 0.5*wh, y + 0.5*w + epsilon]
    lw = patch.Rectangle(xy, wh, ww, angle=np.rad2deg(th), 
                         rotation_point=(x, y), fc='gray', ec='k', alpha=alpha) 
    
    # right wheel
    xy = [x - 0.5*wh, y - 0.5*w - ww - epsilon]
    rw = patch.Rectangle(xy, wh, ww, angle=np.rad2deg(th), 
                        rotation_point=(x, y), fc='gray', ec='k', alpha=alpha) 
    
    return body, lw, rw
"""-------animation-------"""         



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="random")                             # Path Name ("straight", "eight", "change", "random")
    parser.add_argument("--disp_ani", action="store_true")                      # Display animation 
    parser.add_argument("--save_ani", action="store_true")                      # Save animation 
    args = parser.parse_args()    
    print("---------------------------------------")
    print(f"{args.path.capitalize()} Path Following Starts!")
    print("---------------------------------------")
    
    # reference path 
    rpath = ReferencePath(args.path)
    rpath.seed(3) # failure example
    rpath.gen_rand_path()
           
    # vehicle model
    mdl, state = Unicycle(dt=dt), Logger()
    mdl.reset(yaw=rpath.calc_yaw(0.0), flag="random") # re-initialize randomly
    state.append(mdl) # log initial state
    
    # initialize nearest point
    sn = rpath.find_nearest_point(0.0, mdl.x, mdl.y)
    cte, _ = rpath.calc_error(mdl.x, mdl.y, mdl.yaw, sn)
    ls_cte = [cte] # list of cross-track error
    ls_sn = [sn] # list of path parameter for nearest point
    
    # initialize lookahead point
    omega_ref, sl = pure_pursuit_control(mdl, rpath, rpath.calc_lookahead_s0(0.2)) 
    ls_sl = [sl] # list of path parameter for lookahead point
    
    
    while T >= state.t[-1] and \
        rpath.len - sn > 0.02:
        
        # update      
        mdl.update(v_ref, omega_ref)
        #mdl.update_tau(v_ref, omega_ref) 
             
        # next time step
        omega_ref, sl = pure_pursuit_control(mdl, rpath, sl)  
        
        # compute cross-track error
        sn = rpath.find_nearest_point(sn, mdl.x, mdl.y)
        cte, _ = rpath.calc_error(mdl.x, mdl.y, mdl.yaw, sn)
        
        # log
        state.append(mdl)
        ls_sl.append(sl)
        ls_cte.append(cte)
        ls_sn.append(sn)


    # reference path
    s = np.linspace(0.0, rpath.len, 300)
    rx, ry = rpath.X(s), rpath.Y(s)
   
    # compute RMSE of CTE
    cte_pow = [x * x for x in ls_cte]
    rmse = np.sqrt(np.sum(cte_pow) / len(ls_cte))
    print(f"*RMSE: {rmse:.4f}") 
    
    if args.disp_ani:   
        fig, ax = plt.subplots(figsize=(5, 6)) 
        # x-y 
        anim = animation.FuncAnimation(fig, update, interval=1000*dt, frames=len(state.t))
        
        if args.save_ani:
            anim.save('track_ani.gif', writer='pillow')
        
    else:   
        plt.close()
        plt.subplots(1)
        plt.plot(rx, ry, lw=2, color="k", label="reference")
        plt.plot(state.x, state.y, "--", lw=2, color="lime", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        #plt.legend()
        
        plt.subplots(1)
        plt.plot(state.t, state.v, "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("Velocity [m/s]")

        plt.subplots(1)
        plt.plot(state.t, state.omega, "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("Rotational Velocity [deg/s]")
              
        plt.subplots(1)
        plt.plot(state.t, np.rad2deg(state.yaw), "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("Yaw Angle [deg]")
        
        plt.subplots(1)
        plt.plot(state.t, ls_cte, "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("cross-track error [m]")
        
    plt.show()
