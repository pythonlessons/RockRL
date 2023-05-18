# RockRL
Reinforcement Learning library for public, for now, it only supports TensorFlow.

# Installation
```bash
pip install RockRL
```

# Supported Algorithms
- [x] PPO (Discrete and Continuous)

# Examples
- ```RockRL/tensorflow/examples/ppo/LunarLander-v2/LunarLander-v2.py``` is an example of using PPO to solve LunarLander-v2 (Discrete) environment.
- ```RockRL/tensorflow/examples/ppo/BipedalWalker-v3/BipedalWalker-v3.py``` is an example of using PPO to solve BipedalWalker-v2 (Continuous) environment.
- ```RockRL/tensorflow/examples/ppo/BipedalWalker-v3_hardcore/BipedalWalker-v3_hardcore.py``` is an example of using PPO to solve BipedalWalker-v3 Hardcore (Continuous) environment using 4 historical states for future action.
