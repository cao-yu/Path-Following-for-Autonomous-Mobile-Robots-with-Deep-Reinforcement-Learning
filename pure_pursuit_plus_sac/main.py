# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import utils

from algos import SAC
from envs.path_following_env import PathFollowingEnv



# Runs policy for N episodes and returns average reward
def eval_policy(policy, seed, eval_episodes=10):
    eval_env = PathFollowingEnv()
    eval_env.seed(seed + 100)
    
    avg_reward = 0.0
    avg_step = 0.0
    avg_vel = 0.0
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        
        while True:
            #eval_env.render()
            action = policy.select_action(state, deterministic=True) 
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward
            avg_step += 1.0
            
            if terminated or truncated:
                avg_vel += np.mean(eval_env.logger.v)
                break

    avg_reward /= eval_episodes
    avg_step /= eval_episodes   
    avg_vel /= eval_episodes

    print("---------------------------------------") 
    alpha = policy.alpha.cpu().data.numpy().flatten()[0]
    print(f"Evaluation over {eval_episodes} episodes: Alpha: {alpha:.3f}")
    print(f"Avg reward: {avg_reward:.3f} Avg step: {avg_step:.3f} Avg vel: {avg_vel:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_vel, avg_step, alpha




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                        # Policy name (SAC)
    parser.add_argument("--env", default="PathFollowing")                 # Environment name
    parser.add_argument("--path", default="random")                       # Path type ("straight", "eight", "change", "random")
    parser.add_argument("--seed", default=0, type=int)                    # Sets PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e3, type=int)       # Episodes initial random policy is used
    parser.add_argument("--eval_freq", default=25e2, type=int)            # How often (timesteps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5, type=int)         # Max episodes to run environment
    parser.add_argument("--batch_size", default=256, type=int)            # Batch size for both actor and critic
    parser.add_argument("--buffer_capacity", default=5e5, type=int)       # Buffer capacity for replay experience
    parser.add_argument("--noise_type", default="Normal")                 # Type of exploration noise (Normal, OU)
    parser.add_argument("--expl_noise", default=0.1)                      # Std of exploration noise (Normal, OU)
    parser.add_argument("--discount", default=0.99)                       # Discount factor
    parser.add_argument("--tau", default=0.005)                           # Target network update rate
    parser.add_argument("--save_model", action="store_true")              # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                       # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    if not os.path.exists("./train/logs"):
        os.makedirs("./train/logs")

    if args.save_model and not os.path.exists("./train/saved_models"):
        os.makedirs("./train/saved_models")
         
    # load environment
    env = PathFollowingEnv(path=args.path)
            
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    action_space = env.action_space
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "tau": args.tau,
        "action_space": action_space,
        }
        
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)  
    
    # Initialize policy
    policy = SAC.Agent(**kwargs)
        
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./train/saved_models/{policy_file}")  
        
    # Initialize experience replay buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, int(args.buffer_capacity))
    
    # Evaluate policy
    evaluations = [eval_policy(policy, args.seed)]

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    # TRAIN
    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1
        
        # random action or policy action
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state) # original
            #action = (
            #    policy.select_action(state) + noise_obj()
            #    ).clip(action_space.low, action_space.high) 
           
        # perform action 
        next_state, reward, terminated, truncated, _ = env.step(action)
    
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, terminated or truncated)
        
        state = next_state
        episode_reward += reward
        
        # Train policy after collecting sufficient data
        if t >= args.start_timesteps:
            policy.update(replay_buffer, args.batch_size)    
        
        if terminated or truncated:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}, Vel: {np.mean(env.logger.v):.3f}")     
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.seed))
            np.save(f"./train/logs/{file_name}", evaluations) 
  
            if args.save_model:
                policy.save(f"./train/saved_models/{file_name}") # current policy  

    if not os.path.exists(f"./train/logs/{args.policy}_{args.env}_args"):
        np.save(f"./train/logs/{args.policy}_{args.env}_args", args)
    else:
        print("Training arguments have been saved")

    #os.system("shutdown -s -t 60 ")