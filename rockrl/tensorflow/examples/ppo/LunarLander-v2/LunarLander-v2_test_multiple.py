import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gymnasium as gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import models

from rockrl.utils.vectorizedEnv import VectorizedEnv
from rockrl.utils.memory import MemoryManager

if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    num_envs = 4
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name, render_mode="human")

    actor = models.load_model("runs/1696937445/LunarLander-v2_actor.h5", compile=False)
    actor.summary()

    memory = MemoryManager(num_envs=num_envs)
    states, _ = env.reset()
    episodes, episode, score = 100, 0, 0
    while True:

        probs = actor.predict(states, verbose=False)
        actions = np.argmax(probs, axis=-1)

        # state, reward, terminated, truncated, info = env.step(action)
        states, rewards, terminateds, truncateds, infos = env.step(actions)
        env.render()

        memory.append(states, actions, rewards, probs, terminateds, truncateds, states, infos)

        for index in memory.done_indices():
            env_memory = memory[index]
            print(episode, env_memory.score, len(env_memory))

            env_memory.reset()
            states[index], _ = env.reset(index)

            episode += 1

        if episode >= episodes:
            break

    env.close()