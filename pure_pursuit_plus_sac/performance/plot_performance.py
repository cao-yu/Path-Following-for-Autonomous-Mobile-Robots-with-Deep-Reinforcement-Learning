# -*- coding: utf-8 -*-
#
# ATTENTION ！
# In dissertation, straight lines are omitted in performance test.
# For running the script
# go to "param_fcn.py"
# comment the following code
#
# if self.ep_cnt % 10 == 0:
#
# to pass the generation of straight line
#

import sys
from pathlib import Path
base = Path(__file__).resolve()
for i in range(3):
    p = base.parents[i]
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from algos import SAC
from envs.path_following_env import PathFollowingEnv


T = 400.0  # max simulation steps
num = 1000 # Number of attempts at each speed

# cross-track error
cte_tol = np.linspace(0.1, 0.3, 9)


if __name__ == "__main__":
    # run following script
    if 0:
 
        # count failure number
        failure_num = np.zeros((5, len(cte_tol)))
        
        # count completation rate
        completion_rate = np.zeros((num, 5, len(cte_tol)))
        
        # average velocity
        avg_vel = np.zeros((num, 5, len(cte_tol)))
         
        for seed in range(5):
            # load policy
            file_name = f"SAC_PathFollowing_{seed}"
            
            # load environment
            env = PathFollowingEnv("random")
            env.seed(100)
                    
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
            policy.load(f"../train/saved_models/{file_name}") 
        
            for j in tqdm(range(num)):
                state, _ = env.reset() 
                cnt = 0
                k_cte = 0
                
                while True: 
                    
                    cnt += 1
                        
                    action = policy.select_action(state, deterministic=True)           
                    state, reward, terminated, truncated, _ = env.step(action)
                    
                    # terminated
                    if abs(state[0]) >= cte_tol[k_cte]: 
                        failure_num[seed][k_cte] += 1
                        completion_rate[j][seed][k_cte] = env.path.sn / env.path.path.len
                        avg_vel[j][seed][k_cte] = np.mean(env.logger.v)
                        k_cte += 1
                        if k_cte == len(cte_tol):
                            break
                    
                    # reach time limit or goal
                    if T <= cnt or env.path.path.len - env.path.sn < 0.02:
                        while k_cte < len(cte_tol):
                            completion_rate[j][seed][k_cte] = env.path.sn / env.path.path.len
                            avg_vel[j][seed][k_cte] += np.mean(env.logger.v)
                            k_cte += 1
                        break
                        
        
        failure_rate = failure_num / num
        np.save("./failure_rate", failure_rate) 
        np.save("./completion_rate", completion_rate)
        np.save("./avg_vel", avg_vel)
    
    # load from saved files
    else: 
        failure_rate = np.load("./failure_rate.npy") 
        completion_rate = np.load("./completion_rate.npy")
        avg_vel = np.load("./avg_vel.npy")
    
    # plotting
    plt.style.use('classic')
    plt.rcParams["font.size"] = 20

    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    fig.patch.set_alpha(0.0)
     
    avg_failure = np.mean(failure_rate, axis=0)
    std_failure = np.std(failure_rate, axis=0) 
    avg_completion = np.mean(np.mean(completion_rate, axis=0), axis=0) # through the seeds, not samples
    std_completion = np.std(np.mean(completion_rate, axis=0), axis=0)  # through the seeds, not samples
    
    ax.plot(cte_tol, avg_failure, 'b-o', linewidth=2.5, markeredgewidth=3, ms=15, label="Failure")
    #ax.errorbar(cte_tol, avg_failure, xerr=0.0, yerr=std_failure)
    #ax.plot([cte_tol-0.02, cte_tol+0.02], [avg_failure-std_failure, avg_failure-std_failure], 'k', linewidth=2.5)
    #ax.plot([cte_tol-0.02, cte_tol+0.02], [avg_failure+std_failure, avg_failure+std_failure], 'k', linewidth=2.5)
    ax.fill_between(cte_tol, avg_failure - std_failure, avg_failure + std_failure, fc="b", ec="white", alpha=0.5) # confidence bands
    
    ax.plot(cte_tol, avg_completion, 'r-o', linewidth=2.5, markeredgewidth=3, ms=15, label="Completion")
    #ax.errorbar(cte_tol, avg_completion, xerr=0.0, yerr=std_completion)
    #ax.plot([cte_tol-0.02, cte_tol+0.02], [avg_completion-std_completion, avg_completion-std_completion], 'k', linewidth=2.5)
    #ax.plot([cte_tol-0.02, cte_tol+0.02], [avg_completion+std_completion, avg_completion+std_completion], 'k', linewidth=2.5)
    ax.fill_between(cte_tol, avg_completion - std_completion, avg_completion + std_completion, fc="r", ec="white", alpha=0.5) # confidence bands
    
    ax.set_xlabel("Threshold of $|e_p|$ [m]", fontsize=20, labelpad=10)
    ax.set_ylabel("Average Rate", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim(0.09, 0.31)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(loc='center right', numpoints=1, fontsize=20)

    plt.savefig('sac_perform.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close(fig)
    
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    fig.patch.set_alpha(0.0)
   
    avg_v = np.mean(np.mean(avg_vel, axis=0), axis=0) # through the seeds, not samples
    std_v = np.std(np.mean(avg_vel, axis=0), axis=0)  # through the seeds, not samples
    
    ax.plot(cte_tol, avg_v, 'g-o', linewidth=2.5, markeredgewidth=3, ms=15, label="Velocity")
    ax.fill_between(cte_tol, avg_v - std_v, avg_v + std_v, fc="g", ec="white", alpha=0.5) # confidence bands
    
    ax.set_xlabel("Threshold of $|e_p|$ [m]", fontsize=20, labelpad=10)
    ax.set_ylabel("Average Velocity [m/s]", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlim(0.09, 0.31)
    ax.set_ylim(0.26, 0.32)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(loc='lower right', numpoints=1, fontsize=20)
    
    plt.savefig('sac_v_perform.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close(fig)
   