# RockRL
Reinforcement Learning library for public, for now, it only supports **TensorFlow**.

# Installation
```bash
pip install rockrl
```

# Environment requirements
RL algorithms are implemented to support `gymnasium==0.29.1` version. Main requirements are that:
- `env.reset()` would return `state` and `info` states.
- `env.step(action)` would return `state`, `reward`, `terminated`, `truncated`, `info` states.

# Supported Algorithms
- [x] PPO (Discrete and Continuous)

# Code Examples
## Proximal Policy Optimization (PPO):
- [```RockRL/tensorflow/examples/ppo/LunarLander-v2/LunarLander-v2.py```](https://github.com/pythonlessons/RockRL/tree/main/rockrl/tensorflow/examples/ppo/LunarLander-v2) is an example of using PPO to solve LunarLander-v2 (Discrete) environment.
- [```RockRL/tensorflow/examples/ppo/BipedalWalker-v3/BipedalWalker-v3.py```](https://github.com/pythonlessons/RockRL/tree/main/rockrl/tensorflow/examples/ppo/BipedalWalker-v3) is an example of using PPO to solve BipedalWalker-v2 (Continuous) environment.
- [```RockRL/tensorflow/examples/ppo/BipedalWalkerHardcore-v3/BipedalWalkerHardcore-v3.py```](https://github.com/pythonlessons/RockRL/tree/main/rockrl/tensorflow/examples/ppo/BipedalWalkerHardcore-v3) is an example of using PPO to solve BipedalWalker-v3 Hardcore (Continuous) environment.
