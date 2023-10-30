# LunarLander-v2 with PPO and OpenAI Gym (TensorFlow) example
This is an implementation of [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) with OpenAI Gym, using TensorFlow. It is tested on LunarLander-v2 (discrete) environment.

## Usage:
Trained agents are not included in this repository (pretty easy to train by yourself). I recommend training with `LunarLander-v2.py` and testing with `LunarLander-v2_test_multiple.py` (multiple threads) or `LunarLander-v2_test.py` (single thread).
Use as many threads as you can (I recommend 8 or more).

## Requirements (tested versions):
- tensorflow>=2.10.0
- tensorflow_probability>=0.19.0
- gymnasium>=0.29.1
- gymnasium[box2d]>=0.29.1
- rockrl==0.0.4

## Files explanation:
- `LunarLander-v2_single.py` - PPO TensorFlow agent trained with single thread
- `LunarLander-v2_test.py` - Trained agent test with single thread
- `LunarLander-v2.py` - PPO TensorFlow agent trained with multiple threads
- `LunarLander-v2_test_multiple.py` - Trained agent test with multiple threads

## This tutorial YouTube video:
[![LunarLander-v2 with PPO and OpenAI Gym (TensorFlow) example](https://img.youtube.com/vi/OfW78W3Vv2M/0.jpg)](https://www.youtube.com/watch?v=OfW78W3Vv2M)