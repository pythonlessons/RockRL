# BipedalWalker-v3 with PPO and OpenAI Gym (TensorFlow) example
This is an implementation of [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) with OpenAI Gym, using TensorFlow. It is tested on BipedalWalker-v3 (continuous) environment.

## Usage:
Trained agents are not included in this repository (pretty easy to train by yourself).
Use as many workers as you can (I recommend 8 or more).

## Requirements (tested versions):
- tensorflow>=2.10.0
- tensorflow_probability>=0.19.0
- gymnasium>=0.29.1
- gymnasium[box2d]>=0.29.1
- rockrl==0.0.4

## Files explanation:
- `BipedalWalker-v3.py` - PPO TensorFlow agent trained with multiple workers
- `BipedalWalker-v3_test.py` - Trained agent test with multiple workers

## This tutorial YouTube video:
[![BipedalWalker-v3 with PPO and OpenAI Gym (TensorFlow) example](https://img.youtube.com/vi/NNhihM9i1ws/0.jpg)](https://www.youtube.com/watch?v=NNhihM9i1ws)