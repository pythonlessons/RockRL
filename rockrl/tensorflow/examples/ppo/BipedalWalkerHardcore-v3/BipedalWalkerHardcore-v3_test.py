import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gymnasium as gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
from keras.models import load_model

from rockrl.utils.vectorizedEnv import VectorizedEnv
from rockrl.utils.memory import MemoryManager

if __name__ == "__main__":
    env_name = 'BipedalWalkerHardcore-v3'

    num_envs = 4
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name, render_mode="human")
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape

    # actor = load_model("runs/1686313794/BipedalWalkerHardcore-v3_actor.h5", compile=False)
    actor = load_model("runs/1696962888/BipedalWalkerHardcore-v3_actor.h5", compile=False)
    actor.summary()

    memory = MemoryManager(num_envs=num_envs)
    states, _ = env.reset()
    episodes = 100
    episode = 0
    while True:

        probs = actor.predict(states, verbose=False)
        probs_size = int(probs.shape[-1] / 2)
        actions, sigma = probs[:, :probs_size], probs[:, probs_size:]

        next_states, rewards, terminateds, truncateds, infos = env.step(actions)
        memory.append(states, actions, rewards, probs, terminateds, truncateds, next_states, infos)
        states = next_states

        for index in memory.done_indices():
            env_memory = memory[index]
            print(episode, env_memory.score, len(env_memory))

            env_memory.reset()
            states[index], _ = env.reset(index)

            episode += 1

        if episode >= episodes:
            break

    env.close()