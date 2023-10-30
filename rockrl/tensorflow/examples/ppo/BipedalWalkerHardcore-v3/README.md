# BipedalWalker-v3 with PPO and OpenAI Gym (TensorFlow) example
This is an implementation of [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) with OpenAI Gym, using TensorFlow. It is tested on BipedalWalkerHardcore-v3 (continuous) environment.

## Usage:
Trained agents are included in this repository (`1697484446`).
Use as many workers as you can (I recommend 128 or more).

## Requirements (tested versions):
- tensorflow>=2.10.0
- tensorflow_probability>=0.19.0
- gymnasium>=0.29.1
- gymnasium[box2d]>=0.29.1
- rockrl==0.0.4

## Files explanation:
- `BipedalWalkerHardcore-v3.py` - PPO TensorFlow agent trained with multiple workers
- `BipedalWalkerHardcore-v3_test.py` - Trained agent test with multiple workers

## This tutorial YouTube video:
[![BipedalWalkerHardcore-v3 with PPO and OpenAI Gym (TensorFlow) example](https://img.youtube.com/vi/kxMl2Tpil1I/0.jpg)](https://www.youtube.com/watch?v=kxMl2Tpil1I)