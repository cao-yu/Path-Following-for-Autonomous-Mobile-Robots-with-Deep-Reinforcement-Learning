# -*- coding: utf-8 -*-


import argparse
import numpy as np
import matplotlib.pyplot as plt

from kinematics import Unicycle, Logger
from reference_path import ArcReferencePath

from pure_pursuit.control_law import pure_pursuit_control
from pure_pursuit_plus_sac.algos import SAC
from pure_pursuit_plus_sac.envs.path_following_env import PathFollowingEnv


T = 60.0  # max simulation time [s]
dt = 0.05 # [s]
v_ref = 0.4 # [m/s]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="eight")                 # Path Name ("straight", "eight", "change", "random")
    parser.add_argument("--seed", default=0, type=int)             # Sets PyTorch and Numpy seeds
    parser.add_argument("--subplots", default="hor")               # Subplots direction ("hor", "ver")
    args = parser.parse_args()    
    print("---------------------------------------")
    print(f"{args.path.capitalize()}-shaped Path Following Starts!")
    print("---------------------------------------")
    
    """
    PURE PURSUIT
    """
    # reference path 
    rpath = ArcReferencePath(args.path)
    rpath.seed(100)
           
    # vehicle model
    mdl, state = Unicycle(dt=dt), Logger()
    mdl.seed(100)
    mdl.reset(yaw=rpath.calc_yaw(0.0), flag="random") # re-initialize randomly
    state.append(mdl) # log initial state
    
    # initialize nearest and lookahead point
    omega_ref, sn, sl = pure_pursuit_control(mdl, rpath, 0.0) 
    cte, _ = rpath.calc_error(mdl.x, mdl.y, mdl.yaw, sn)
    ls_cte_pp = [cte] # list of cross-track error
    ls_sn_pp = [sn] # list of path parameter for nearest point
    
    while T >= state.t[-1] and \
        rpath.len - sn > 0.02:
        
        # update      
        mdl.update(v_ref, omega_ref)
             
        # next time step
        omega_ref, sn, sl = pure_pursuit_control(mdl, rpath, sn)  
        
        # compute cross-track error
        cte, _ = rpath.calc_error(mdl.x, mdl.y, mdl.yaw, sn)
        
        # log
        state.append(mdl)
        ls_cte_pp.append(cte)
        ls_sn_pp.append(sn)
        
    # compute RMSE of CTE
    cte_pow = [x * x for x in ls_cte_pp]
    rmse = np.sqrt(np.sum(cte_pow) / len(ls_cte_pp))
    print(f"*RMSE: {rmse:.4f}") 
    print(f"*MAX: {max(np.abs(ls_cte_pp)):.4f}") 
    print(f"*time: {state.t[-1]:.2f}")
    print("---------------------------------------")
        
    """
    SAC
    """
    # Initialize policy
    env = PathFollowingEnv(path=args.path)
    env.seed(100)
            
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    action_space = env.action_space
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": action_space,
        }
    policy = SAC.Agent(**kwargs)    
    file_name = f"SAC_PathFollowing_{args.seed}"
    policy.load(f"./pure_pursuit_plus_sac/train/saved_models/{file_name}")  
    

    # initialization
    obs, _ = env.reset()
    ls_cte_sac = [] # list of cross-track error
    ls_sn_sac = [] # list of path parameter for nearest point
       
    while env.path.path.len - env.path.sn > 0.02:  
        # log
        ls_cte_sac.append(obs[0])
        ls_sn_sac.append(env.path.sn)
        
        # step    
        action = policy.select_action(obs, deterministic=True) 
        obs, _, _, _, _ = env.step(action)
        
        if env.logger.t[-1] > T:
            break
        
    cte_pow = [x * x for x in ls_cte_sac]
    rmse = np.sqrt(np.sum(cte_pow) / len(ls_cte_sac))
    print(f"*RMSE: {rmse:.4f}") 
    print(f"*MAX: {max(np.abs(ls_cte_sac)):.4f}") 
    print(f"*avg_vel: {np.mean(env.logger.v[1:]):.4f}")
    print(f"*time: {env.logger.t[-1]:.2f}")
    print("---------------------------------------")
    
    """
    PLOTTING trajectories and cte
    """ 
    # reference path
    s = np.linspace(0.0, rpath.len, 300)
    rx, ry = rpath.X(s), rpath.Y(s)
    
    # subplots 1 x 2
    if args.subplots == "hor":
        fig, ax = plt.subplots(1, 2, figsize=(13., 5.5))
        plt.subplots_adjust(wspace=0.3, top=0.9, bottom=0.13, left=0.085, right=0.99)
        ax[0].text(-0.14, 1.12, '(a)', transform=ax[0].transAxes, fontsize=18, va='top', ha='right')
        ax[1].text(-0.175, 1.12, '(b)', transform=ax[1].transAxes, fontsize=18, va='top', ha='right')

    # subplots 2 x 1
    if args.subplots == "ver":
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        plt.subplots_adjust(hspace=0.25, top=0.89, bottom=0.065, left=0.15, right=0.94)

    plt.rcParams["font.size"] = 18
    plt.style.use('classic')
    fig.patch.set_alpha(0.0)
    

    ax[0].plot(rx, ry, linestyle=(5, (10, 3)), color="k", lw=2.5, label="Reference")
    ax[0].plot(state.x, state.y, color='b', lw=2.5, label="Pure Pursuit")
    ax[0].plot(env.logger.x, env.logger.y, color='r', lw=2.5, label="PP + SAC")
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].set_ylim(-0.6, 0.6)
    ax[0].grid(True)
    ax[0].set_xlabel("x [m]", fontsize=18)
    ax[0].set_ylabel("y [m]", fontsize=18)
    ax[0].tick_params(axis='both', labelsize=18)
    legend = ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=18)
    legend.get_frame().set_alpha(0.5)

    ax[1].plot(ls_sn_pp, ls_cte_pp, color='b', lw=2.5, label="Pure Pursuit")
    ax[1].plot(ls_sn_sac, ls_cte_sac, color='r', lw=2.5, label="PP + SAC")
    ax[1].set_xlim(ls_sn_pp[0], ls_sn_pp[-1])
    ax[1].set_ylim(min(ls_cte_pp)-0.01, max(ls_cte_pp)+0.01)
    ax[1].grid(True)
    ax[1].set_xlabel("Parameter $\lambda$", fontsize=18)
    ax[1].set_ylabel("cross-track error [m]", fontsize=18)
    ax[1].tick_params(axis='both', labelsize=18)
    legend = ax[1].legend(loc='upper right', ncol=1, fontsize=18)
    legend.get_frame().set_alpha(0.5)
    plt.savefig('sim_eight.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
        
    """
    PLOTTING velocities
    """ 
    # subplot 1 x 2
    if args.subplots == "hor":
        fig, ax = plt.subplots(1, 2, figsize=(13., 5.))
        plt.subplots_adjust(wspace=0.3, top=0.9, bottom=0.13, left=0.085, right=0.99)
        ax[0].text(-0.1, 1.12, '(a)', transform=ax[0].transAxes, fontsize=18, va='top', ha='right')
        ax[1].text(-0.14, 1.12, '(b)', transform=ax[1].transAxes, fontsize=18, va='top', ha='right')

    # subplots 2 x 1
    if args.subplots == "ver":
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        plt.subplots_adjust(hspace=0.25, top=0.89, bottom=0.065, left=0.15, right=0.94)

    plt.rcParams["font.size"] = 18
    plt.style.use('classic')
    fig.patch.set_alpha(0.0)

    ax[0].plot(ls_sn_pp, state.v, color='b', lw=2.5, label="Pure Pursuit")
    ax[0].plot(ls_sn_sac, env.logger.v, color='r', lw=2.5, label="PP + SAC")
    ax[0].set_xlim(ls_sn_pp[0], ls_sn_pp[-1])
    ax[0].set_ylim(0.0, 0.41)
    ax[0].grid(True)
    ax[0].set_xlabel("Parameter $\lambda$", fontsize=18)
    ax[0].set_ylabel("Linear Velocity [m/s]", fontsize=18)
    ax[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax[0].tick_params(axis='both', labelsize=18)
    legend = ax[0].legend(loc='lower right', ncol=1, fontsize=18)
    legend.get_frame().set_alpha(0.5)
    
    ax[1].plot(ls_sn_pp, state.omega, color='b', lw=2.5, label="Pure Pursuit")
    ax[1].plot(ls_sn_sac, env.logger.omega, color='r', lw=2.5, label="PP + SAC")
    ax[1].set_xlim(ls_sn_pp[0], ls_sn_pp[-1])
    ax[1].set_ylim(-1.05, 1.05)
    ax[1].grid(True)
    ax[1].set_xlabel("Parameter $\lambda$", fontsize=18)
    ax[1].set_ylabel("Angular Velocity [rad/s]", fontsize=18)
    ax[1].tick_params(axis='both', labelsize=18)
    legend = ax[1].legend(loc='lower right', ncol=1, fontsize=18)
    legend.get_frame().set_alpha(0.5)
    plt.savefig('sim_eight_vel.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    
    """
    PLOTTING linear velocity on trajectory
    """ 
    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(1, figsize=(8, 5.))
    plt.style.use('classic')
    fig.patch.set_alpha(0.0)
    plt.subplots_adjust(wspace=0.4, bottom=0.13, top=0.9)
    
    # create a scatterplot, setting the colormap to 'jet'
    ax.plot(rx, ry, linestyle=(5, (10, 3)), color="k", lw=2.5, label="Reference")
    sc = plt.scatter(env.logger.x, env.logger.y, s=80, c=env.logger.v, edgecolors='face', cmap='jet')

    # colorbar
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=18) 
    cbar.set_label('Velocity Values [m/s]', fontsize=18)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True)
    ax.set_xlabel("x [m]", fontsize=18)
    ax.set_ylabel("y [m]", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    plt.savefig('sim_eight_vel_on_traj.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.show()