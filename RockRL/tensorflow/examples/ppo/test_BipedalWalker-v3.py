import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.models import Model, load_model

from RockRL.utils.vectorizedEnv import VectorizedEnv, CustomEnv
from RockRL.utils.misc import MeanAverage
from RockRL.utils.memory import Memory


if __name__ == "__main__":
    env_name = 'BipedalWalker-v3'

    num_envs = 6
    env = VectorizedEnv(env_object=CustomEnv, custom_env_object=gym.make, os_hist_steps=4, num_envs=num_envs, id=env_name, render_mode="human", hardcore=True)
    # env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name) # , render_mode="human")
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape

    actor = load_model("runs/1684128290/BipedalWalker-v3_actor.h5", compile=False)
    actor.summary()

    memory = Memory(num_envs=num_envs, input_shape=input_shape)
    # meanAverage = MeanAverage(
    #     window_size=50,
    #     best_mean_score=-np.inf,
    #     best_mean_score_episode=500,
    # )
    states = env.reset()
    episodes = 10000
    episode = 0
    while True:

        probs = actor.predict(states, verbose=False)
        # probs = agent(states).numpy()
        probs_size = int(probs.shape[-1] / 2)
        actions, sigma = probs[:, :probs_size], probs[:, probs_size:]
        # noise = np.random.normal(0, 0.1, size=sigma.shape)
        # actions = np.random.normal(a_probs, sigma)
        # actions += noise
        # actions = np.clip(actions, -1, 1)

        # actions, probs = agent.act(states)

        next_states, rewards, dones = env.step(actions)
        memory.append(states, actions, rewards, probs, dones, next_states)
        states = next_states

        rewards_len = np.array([len(r) for r in memory.rewards])
        if np.any(rewards_len >= env._max_episode_steps):
            dones = np.array([True if r >= env._max_episode_steps else False for r in rewards_len])

        for index in np.where(dones)[0]:
            _states, _actions, _rewards, _predictions, _dones, _next_state = memory.get(index=index)
            # agent.train(_states, _actions, _rewards, _predictions, _dones, _next_state)
            memory.reset(index=index)
            states[index] = env.reset(index)

            episode += 1
            score = np.sum(_rewards)
            print(episode, score, len(_rewards))
            # mean = meanAverage(score)

            # if meanAverage.is_best(episode):
            #     # save model
            #     agent.save_models(env_name)

            # if not meanAverage.is_improoving(episode):
            #     agent.reduce_learning_rate(ratio=0.95, verbose=True)

            # print(episode, score, mean, len(_rewards), meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if episode >= episodes:
            break

    env.close()