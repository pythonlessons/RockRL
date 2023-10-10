import os
import gymnasium as gym
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # visible only one gpu
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import models, layers

from rockrl.utils.vectorizedEnv import VectorizedEnv
from rockrl.utils.misc import MeanAverage
from rockrl.utils.memory import Memory
from rockrl.tensorflow import PPOAgent

def actor_model(input_shape, action_space):
    X_input = layers.Input(input_shape)
    X = layers.Dense(512)(X_input)
    X = layers.LeakyReLU(alpha=0.1)(X)
    X = layers.Dense(256)(X)
    X = layers.LeakyReLU(alpha=0.1)(X)
    X = layers.Dense(64)(X)
    X = layers.LeakyReLU(alpha=0.1)(X)

    action = layers.Dense(action_space, activation="tanh")(X)
    sigma = layers.Dense(action_space, activation='softplus')(X)
    action_sigma = layers.concatenate([action, sigma])

    model = models.Model(inputs = X_input, outputs = action_sigma)
    return model

def critic_model(input_shape):
    X_input = layers.Input(input_shape)
    X = layers.Dense(512)(X_input)
    X = layers.LeakyReLU(alpha=0.1)(X)
    X = layers.Dense(256)(X)
    X = layers.LeakyReLU(alpha=0.1)(X)
    X = layers.Dense(64)(X)
    X = layers.LeakyReLU(alpha=0.1)(X)

    value = layers.Dense(1, activation=None)(X)

    model = models.Model(inputs = X_input, outputs = value)
    return model


if __name__ == "__main__":
    env_name = 'BipedalWalkerHardcore-v3'

    num_envs = 48
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name)
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        actor = actor_model(input_shape, action_space),
        critic = critic_model(input_shape),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        action_space="continuous",
        batch_size=512,
        train_epochs=10,
        gamma=0.99,
        lamda=0.90,
        c2=0.01,
        kl_coeff=0.2,
        shuffle=True,
        writer_comment=env_name,
    )
    agent.actor.summary()

    memory = Memory(num_envs=num_envs, input_shape=input_shape)
    meanAverage = MeanAverage(window_size=100)
    states = env.reset()
    episodes = 500000
    episode = 0
    while True:
        actions, probs = agent.act(states)

        next_states, rewards, dones, _ = env.step(actions)
        memory.append(states, actions, rewards, probs, dones, next_states)
        states = next_states

        for index in memory.done_indices(env._max_episode_steps):
            _states, _actions, _rewards, _predictions, _dones, _next_state = memory.get(index=index)
            agent.train(_states, _actions, _rewards, _predictions, _dones, _next_state)
            memory.reset(index=index)
            states[index] = env.reset(index)

            episode += 1
            score = np.sum(_rewards)
            mean = meanAverage(score)

            if meanAverage.is_best(episode):
                # save model
                agent.save_models(env_name)

            print(episode, score, mean, len(_rewards), meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if episode >= episodes:
            break

    env.close()