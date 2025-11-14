# DRL-Autonomous-Path-Following

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs24020561-blue)](https://doi.org/10.3390/s24020561)

This repository contains the source code for the paper "Path Following for Autonomous Mobile Robots with Deep Reinforcement Learning".

## 🚀 Highlights

- **DRL-augmented pure pursuit controller** for autonomous mobile robot path following.  
- **Adaptive to local path geometry**: the policy reacts to current path curvature and the lookahead point, reducing cross-track error.  
- **Robust tracking performance** in simulation, with smooth velocity and steering profiles.  
- Modular codebase for **easy reproduction and extension** (new environments, reward functions, or robot models).

## 🎮 DRL-augmented Pure Pursuit Path Following

<p align="center">
  <img src="assets/ani_eight.gif" alt="Eight-shaped Path Following" width="100%">
  <img src="assets/ani_change.gif" alt="Lane-change Path Following" width="100%">
</p>

The two animations above illustrate how our DRL-augmented pure pursuit controller tracks different reference paths:

- **Left – Eight-shaped path:** the controller slows down in high-curvature regions and accelerates on straighter segments, while keeping the cross-track error small.  
- **Right – Lane-change path:** the controller anticipates the upcoming lateral shift using the lookahead point and smoothly adjusts steering and velocity to execute the lane change.

In both cases, the trajectory is color-coded by **linear velocity**, showing how the policy adapts the speed profile according to the local path curvature and future path segment around the lookahead point.

## 🌐 Generalization to Random Paths

<p align="center">
  <img src="assets/rand_pf.jpg" alt="Random Path Following" width="100%">
</p>

To test generalization, we also evaluate the controller on **randomly generated smooth paths** that are not seen during training.

<p align="center">
  <img src="assets/random_paths.jpg" alt="Random Path Following" width="70%">
</p>

Even on these unseen paths, the DRL-augmented pure pursuit controller is able to:

- adapt its velocity to local curvature,
- keep the robot close to the reference,
- and maintain smooth, collision-free trajectories.

This suggests that the learned policy does not simply memorize a small set of trajectories, but instead **learns a control strategy that transfers to new path geometries**.
