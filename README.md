# DRL-Autonomous-Path-Following

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs24020561-blue)](https://doi.org/10.3390/s24020561)

This repository contains the source code for the paper **“Path Following for Autonomous Mobile Robots with Deep Reinforcement Learning”**.

## 🚀 Highlights

- **Pure Pursuit + SAC**: Pure Pursuit handles steering, while a SAC-based policy adaptively controls the linear velocity.  
- **Path-relative speed control**: The policy uses the robot’s state relative to the path (including the `nearest` and `lookahead` points) to adjust speed and reduce cross-track error.  
- **Generalization to unseen paths**: The same controller works on randomly generated paths without retraining, showing strong path-level generalization.  

## 🎮 Pure Pursuit Steering + SAC Velocity Control

These animations show our **Pure Pursuit + SAC controller** in action. The trajectory is color-coded by **linear velocity**, highlighting how the learned policy adjusts speed based on the robot’s state relative to the path (including the `nearest` and `lookahead` points).

<p align="center">
  <img src="assets/ani_eight.gif" alt="Eight-shaped Path Following" width="70%">
</p>

- **Eight-shaped path:** Pure Pursuit steers the robot, while SAC adapts the speed using path-relative information from the `nearest` and `lookahead` points, slowing in demanding segments and accelerating where tracking is easier.  

<p align="center">
  <img src="assets/ani_change.gif" alt="Lane-change Path Following" width="70%">
</p>

- **Lane-change path:** Using both the `nearest` and `lookahead` points, SAC shapes the velocity profile so that Pure Pursuit can anticipate the lateral shift and execute the lane change smoothly with small tracking error.

## 🌐 Generalization to Random Paths

To test generalization, we evaluate the **Pure Pursuit + SAC controller** on randomly generated smooth paths that are not seen during training.  

<p align="center">
  <img src="assets/rand_pf.jpg" alt="Random Path Following" width="100%">
</p>

The starting point is randomly generated in the vicinity of (0, 0).
