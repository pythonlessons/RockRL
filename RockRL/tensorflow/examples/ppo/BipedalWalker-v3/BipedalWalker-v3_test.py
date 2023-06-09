import os
import gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.models import Model, load_model

from RockRL.utils.vectorizedEnv import VectorizedEnv
from RockRL.utils.memory import Memory


if __name__ == "__main__":
    env_name = 'BipedalWalker-v3'

    num_envs = 4
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name, render_mode="human")
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape

    actor = load_model("runs/1686059965/BipedalWalker-v3_actor.h5", compile=False)
    actor.summary()

    memory = Memory(num_envs=num_envs, input_shape=input_shape)
    states = env.reset()
    episodes = 10000
    episode = 0
    while True:

        probs = actor.predict(states, verbose=False)
        probs_size = int(probs.shape[-1] / 2)
        actions, sigma = probs[:, :probs_size], probs[:, probs_size:]

        next_states, rewards, dones = env.step(actions)
        memory.append(states, actions, rewards, probs, dones, next_states)
        states = next_states

        for index in memory.done_indices(env._max_episode_steps):
            _states, _actions, _rewards, _predictions, _dones, _next_state = memory.get(index=index)
            memory.reset(index=index)
            states[index] = env.reset(index)

            episode += 1
            score = np.sum(_rewards)
            print(episode, score, len(_rewards))

        if episode >= episodes:
            break

    env.close()