# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                        # Policy name (SAC)
    parser.add_argument("--env", default="PathFollowing")                 # Environment name
    args = parser.parse_args()
    args_file = np.load(f"./logs/{args.policy}_{args.env}_args.npy", allow_pickle=True).item()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}")
    print("---------------------------------------")

    # compute avg return/velocity per trial
    reward, vel = [], []
    start_index = 0

    seed_arr = np.arange(5)
    #seed_arr = np.array([3]) # check individual files
    for seed in seed_arr: 
        file_name = f"{args.policy}_{args.env}_{seed}"
        evaluations = np.load(f"./logs/{file_name}.npy")  
    
        if start_index == 0:
            start_index = int(np.ceil(args_file.start_timesteps/args_file.eval_freq))

        reward = np.append(reward, evaluations[start_index:, 0]) # average return
        vel = np.append(vel, evaluations[start_index:, 1]) # average velocity

    reward = reward.reshape(len(seed_arr), -1)
    mu_rew = np.average(reward, axis=0) 
    sigma_rew = np.std(reward, axis=0) 

    vel = vel.reshape(len(seed_arr), -1)
    mu_v = np.average(vel, axis=0)
    sigma_v = np.std(vel, axis=0)

    t =  np.linspace(0, 5, len(mu_rew))

    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use("default")
    fig.patch.set_alpha(0.0)
    plt.subplots_adjust(wspace=0.25, top=0.905, bottom=0.16, left=0.105, right=0.975)
    ax[0].text(-0.17, 1.12, '(a)', transform=ax[0].transAxes, fontsize=18, va='top', ha='right')
    ax[1].text(-0.11, 1.12, '(b)', transform=ax[1].transAxes, fontsize=18, va='top', ha='right')

    ax[0].plot(t, mu_rew, color="tab:blue", lw=2.)
    ax[0].fill_between(t, mu_rew - sigma_rew, mu_rew +  sigma_rew, fc="tab:blue", ec="white", alpha=0.5) # confidence bands
    ax[0].set_xlim(0, 5)
    ax[0].set_ylim(-200, 250)
    ax[0].set_xlabel("Time Steps " + r"(1 x $10^5$)", fontsize=18)
    ax[0].set_ylabel("Average Return", fontsize=18)
    ax[0].tick_params(axis='both', labelsize=18)
    ax[0].grid(True, linestyle='dashed')  
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    
    ax[1].plot(t, mu_v, color="tab:orange", lw=2.)
    ax[1].fill_between(t, mu_v - sigma_v, mu_v +  sigma_v, fc="tab:orange", ec="white", alpha=0.5) # confidence bands
    ax[1].set_xlim(0, 5)
    ax[1].set_ylim(0, 0.37)
    ax[1].set_xlabel("Time Steps " + r"(1 x $10^5$)", fontsize=18)
    ax[1].set_ylabel("Average Velocity [m/s]", fontsize=18)
    ax[1].tick_params(axis='both', labelsize=18)
    ax[1].grid(True, linestyle='dashed')  
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.show()
    #plt.tight_layout()
    #plt.savefig('learning_curve.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.05)








