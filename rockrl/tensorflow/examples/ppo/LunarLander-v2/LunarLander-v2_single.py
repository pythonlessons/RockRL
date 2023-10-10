import os
import gymnasium as gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import models, layers

from rockrl.utils.misc import MeanAverage
from rockrl.utils.memory import Memory
from rockrl.tensorflow import PPOAgent


def actor_model(input_shape, action_space):
    X_input = layers.Input(input_shape)
    X = layers.Dense(512, activation='relu')(X_input)
    X = layers.Dense(256, activation='relu')(X)
    X = layers.Dense(64, activation='relu')(X)
    output = layers.Dense(action_space, activation="softmax")(X)

    model = models.Model(inputs = X_input, outputs = output)
    return model

def critic_model(input_shape):
    X_input = layers.Input(input_shape)
    X = layers.Dense(512, activation="relu")(X_input)
    X = layers.Dense(256, activation="relu")(X)
    X = layers.Dense(64, activation="relu")(X)
    value = layers.Dense(1, activation=None)(X)

    model = models.Model(inputs = X_input, outputs = value)
    return model


if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    env = gym.make(env_name)
    action_space = env.action_space.n
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        actor = actor_model(input_shape, action_space),
        critic = critic_model(input_shape),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        batch_size=512,
        kl_coeff=0.2,
        c2=0.1,
        writer_comment=env_name,
    )

    memory = Memory()
    meanAverage = MeanAverage()
    state, info = env.reset()
    episodes = 10000
    while True:
        action, prob = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        memory.append(state, action, reward, prob, terminated, truncated, next_state, info)
        state = next_state

        if memory.done:
            agent.train(memory)
            state, info = env.reset()

            score = np.sum(memory.rewards)
            mean = meanAverage(score)

            if meanAverage.is_best(agent.epoch):
                # save best model
                agent.save_models(env_name)

            print(agent.epoch, score, mean, len(memory.rewards), meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if agent.epoch >= episodes:
            break

    env.close()