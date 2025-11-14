# -*- coding: utf-8 -*-
# python eval_policy.py --policy SAC --eval_episodes 1 --load_model default --disp_ani

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patch

from algos import SAC
from envs.path_following_env import PathFollowingEnv


cbar = None

"""-------animation-------"""
def update(i):  
    global cbar

    # frame initialization
    ax.cla()
    plt.rcParams["font.size"] = 12
    plt.subplots_adjust(top=0.98, bottom=0.05, left=0.1, right=1)

    ax.set_xlim(min(rx) - 0.2, max(rx) + 0.2)
    ax.set_ylim(min(ry) - 0.2, max(ry) + 0.2)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True) 
    plt.title(
        f"t:{env.logger.t[i]:.2f} s, v:{env.logger.v[i]:.2f} m/s, "
        r"$\omega$" f":{env.logger.omega[i-1]:.2f} rad/s"
    )
    
    # reference path
    plt.plot(rx, ry, 'k', lw=3, label="Reference")
    
    # nearest point
    plt.plot(rpath.X(ls_sn[i]), rpath.Y(ls_sn[i]), 'bo', markeredgewidth=3, ms=10, label="Nearest Point")
    
    # lookahead point
    plt.plot(rpath.X(ls_sl[i]), rpath.Y(ls_sl[i]), 'rx', markeredgewidth=3, ms=10, label="Lookahead Point")

    # trajectory
    #plt.plot(env.logger.x[0: i+1], env.logger.y[0: i+1], '--', lw=3, color='lime', label="Trajectory")

    # ===== trajectory with velocity color =====
    sac_x = env.logger.x[0: i+1]
    sac_y = env.logger.y[0: i+1]
    sac_v = env.logger.v[0: i+1]

    sc = plt.scatter(
        sac_x, sac_y,
        s=60,
        c=sac_v,
        edgecolors='face',
        cmap='jet',
        vmin=0.0,
        vmax=env.mdl.max_v
    )

    # Add colorbar only once
    if cbar is None:
        cbar = ax.figure.colorbar(sc, ax=ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Velocity [m/s]', fontsize=12)
    # ==========================================

    # eight
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 0.98),  
        ncol=3, frameon=True, fontsize=10
    )

    # change
    #ax.legend( 
    #    loc='upper left', ncol=1,                      
    #    frameon=True, fontsize=10
    #)

    # robot
    body, leftw, rightw = plot_robot(
        env.logger.x[i], env.logger.y[i], env.logger.yaw[i],
        clr='b', alpha=0.5
    )
    bd = ax.add_patch(body)
    lw = ax.add_patch(leftw)
    rw = ax.add_patch(rightw)
      
    return bd, lw, rw,
   

def plot_robot(x, y, th, clr, w=0.172, h=0.2, wh=0.15, ww=0.03, alpha=None):
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                        # Policy name (SAC)
    parser.add_argument("--env", default="PathFollowing")                 # Environment name
    parser.add_argument("--path", default="random")                       # Path type ("straight", "eight", "change", "random")
    parser.add_argument("--seed", default=0, type=int)                    # Sets PyTorch and Numpy seeds
    parser.add_argument("--eval_episodes", default=1, type=int)           # Episode number of evalution 
    parser.add_argument("--load_model", default="")                       # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--disp_ani", action="store_true")                # Display animation 
    parser.add_argument("--save_ani", action="store_true")                # Save animation 
    args = parser.parse_args()
    
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    # load environment
    env = PathFollowingEnv(path=args.path)
            
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    action_space = env.action_space
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": action_space,
        }

    # Initialize policy
    policy = SAC.Agent(**kwargs)
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./train/saved_models/{file_name}")  

    # Evaluation
    avg_reward = 0.0
    avg_step = 0.0
    avg_vel = 0.0

    for ep in range(args.eval_episodes):
        ls_cte = [] # list of cross-track error
        ls_sn = [] # list of path parameter for nearest point
        ls_sl = [] # list of path parameter for lookahead point

        state, _ = env.reset(T=0.0, k=0) 
        
        while True:        
            if args.disp_ani:
                env.render()     

            # log for saving animation
            ls_cte.append(state[0])
            ls_sn.append(env.path.sn)
            ls_sl.append(env.path.sl)

            # step forward
            action = policy.select_action(state, deterministic=True) 
            state, reward, terminated, truncated, _ = env.step(action)
            avg_reward += reward
            avg_step += 1.0
            
            #if terminated or truncated:
            if env.path.check_truncation():
                avg_vel += np.mean(env.logger.v)
                break

        # compute RMSE of CTE
        cte_pow = [x * x for x in ls_cte]
        rmse = np.sqrt(np.sum(cte_pow) / len(ls_cte))
        print(f"*RMSE of Episode {ep}: {rmse:.4f}") 

    avg_reward /= args.eval_episodes
    avg_step /= args.eval_episodes
    avg_vel /= args.eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {args.eval_episodes} episodes: ")
    print(f"Avg reward: {avg_reward:.3f} Avg step: {avg_step:.3f} Avg vel: {avg_vel:.3f}")
    print("---------------------------------------")

    #--------------------------------------------------------------------------------------
    # PLOT OR SAVE ANIMATION
    #--------------------------------------------------------------------------------------
    # reference path
    rpath = env.path.path
    s = np.linspace(0.0, rpath.len, 300)
    rx, ry = rpath.X(s), rpath.Y(s)
    
    if args.save_ani:   
        fig, ax = plt.subplots(figsize=(9, 5)) 
        anim = animation.FuncAnimation(
            fig, 
            update, 
            interval=1000*env.mdl.dt, 
            frames=len(env.logger.t),
            blit=False
            )
        anim.save(f"ani_{args.path}.gif", writer='pillow')
        plt.close(fig)
        
    elif not args.disp_ani and args.eval_episodes < 2:   
        plt.figure()
        plt.plot(rx, ry, lw=2, color="k", label="reference")
        plt.plot(env.logger.x, env.logger.y, "--", lw=2, color="lime", label="trajectory")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend()
        
        plt.figure()
        plt.plot(env.logger.t, env.logger.v, "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("Velocity [m/s]")

        plt.figure()
        plt.plot(env.logger.t, np.rad2deg(env.logger.omega), "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("Rotational Velocity [deg/s]")
              
        plt.figure()
        plt.plot(env.logger.t, np.rad2deg(env.logger.yaw), "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("Yaw Angle [deg]")
        
        plt.figure()
        plt.plot(env.logger.t, ls_cte, "-r", lw=2)
        plt.grid(True)
        plt.xlabel("time [s]")
        plt.ylabel("cross-track error [m]")
        
        plt.show()
