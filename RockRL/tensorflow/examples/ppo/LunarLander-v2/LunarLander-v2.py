import os
import gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.models import Model
from keras.layers import Input, Dense

from RockRL.utils.vectorizedEnv import VectorizedEnv
from RockRL.utils.misc import MeanAverage
from RockRL.utils.memory import Memory
from RockRL.tensorflow import PPOAgent


def actor_model(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, activation='relu')(X_input)
    X = Dense(256, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    output = Dense(action_space, activation="softmax")(X)

    model = Model(inputs = X_input, outputs = output)
    return model

def critic_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(512, activation="relu")(X_input)
    X = Dense(256, activation="relu")(X)
    X = Dense(64, activation="relu")(X)
    value = Dense(1, activation=None)(X)

    model = Model(inputs = X_input, outputs = value)
    return model


if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    num_envs = 24
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name) # , render_mode="human")
    action_space = env.action_space.n
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        actor = actor_model(input_shape, action_space),
        critic = critic_model(input_shape),
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        batch_size=256
    )

    memory = Memory(num_envs=num_envs, input_shape=input_shape)
    meanAverage = MeanAverage(
        window_size=50,
        best_mean_score=-np.inf,
        best_mean_score_episode=100,
    )
    wait_best_window = 50
    reduce_lr_episode = 100
    states = env.reset()
    episodes = 10000
    episode = 0
    while True:
        actions, probs = agent.act(states)

        next_states, rewards, dones = env.step(actions)
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

                if meanAverage.best_mean_score_episode > reduce_lr_episode:
                    reduce_lr_episode = meanAverage.best_mean_score_episode + wait_best_window

            if reduce_lr_episode < episode:
                agent.reduce_learning_rate(ratio=0.95, verbose=True, min_lr = 5e-06)
                reduce_lr_episode = episode + wait_best_window

            print(episode, score, mean, len(_rewards), meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if episode >= episodes:
            break

    env.close()