# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)

import argparse
import torch as th

from algos import SAC
from envs.path_following_env import PathFollowingEnv

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                        # Policy name (SAC)
    parser.add_argument("--env", default="PathFollowing")                 # Environment name
    parser.add_argument("--seed", default=0, type=int)                    # Sets PyTorch and Numpy seeds
    args = parser.parse_args()
    
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    # load environment
    env = PathFollowingEnv()
            
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
    policy_file = file_name 
    policy.load(f"./saved_models/{policy_file}")  
    policy.actor.eval()

    dummy_input = th.randn(1, state_dim).to(th.device("cuda" if th.cuda.is_available() else "cpu"))
    th.onnx.export(
        policy.actor,
        dummy_input,
        f"sac_actor_{args.seed}.onnx",
        opset_version=10,
        input_names=["input"],
    )
    print(policy.actor(dummy_input))
    
    ##### Load and test with onnx

    import onnxruntime as ort
    #import numpy as np

    onnx_path = f"sac_actor_{args.seed}.onnx"

    #observation = np.zeros((1, state_dim)).astype(np.float32)
    observation = dummy_input.cpu().data.numpy()
    ort_sess = ort.InferenceSession(onnx_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    action = ort_sess.run(None, {"input": observation})
    print(action)
    
    # 0.35 * np.tanh(action[0]) - 0.05