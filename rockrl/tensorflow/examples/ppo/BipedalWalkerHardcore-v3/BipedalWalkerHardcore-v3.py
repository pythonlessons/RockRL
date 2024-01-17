import os
import gymnasium as gym
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # visible only one gpu
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import models, layers

from rockrl.utils.vectorizedEnv import VectorizedEnv
from rockrl.utils.misc import MeanAverage
from rockrl.utils.memory import MemoryManager
from rockrl.tensorflow import PPOAgent

def actor_model(input_shape, action_space):
    X_input = layers.Input(input_shape)
    X = layers.Dense(512, activation='relu')(X_input)
    X = layers.Dense(256, activation='relu')(X)
    X = layers.Dense(64, activation='relu')(X)

    action = layers.Dense(action_space, activation="tanh")(X)
    sigma = layers.Dense(action_space)(X)
    sigma = layers.Dense(1, activation='sigmoid')(sigma)
    action_sigma = layers.concatenate([action, sigma])

    model = models.Model(inputs = X_input, outputs = action_sigma)
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
    env_name = 'BipedalWalkerHardcore-v3'

    num_envs = 128
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name)
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        actor = actor_model(input_shape, action_space),
        critic = critic_model(input_shape),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        action_space="continuous",
        batch_size=512,
        train_epochs=10,
        gamma=0.99,
        lamda=0.90,
        c2=0.01,
        kl_coeff=0.5,
        writer_comment=env_name,
    )
    agent.actor.summary()

    memory = MemoryManager(num_envs=num_envs)
    meanAverage = MeanAverage(window_size=100)
    states, _ = env.reset()
    episodes = 500000
    while True:
        actions, probs = agent.act(states)

        next_states, rewards, terminateds, truncateds, infos = env.step(actions)
        memory.append(states, actions, rewards, probs, terminateds, truncateds, next_states, infos)
        states = next_states

        for index in memory.done_indices():
            env_memory = memory[index]
            agent.train(env_memory)
            states[index], _ = env.reset(index)

            mean = meanAverage(env_memory.score)

            if meanAverage.is_best(agent.epoch):
                # save model
                agent.save_models(env_name)

            print(agent.epoch, env_memory.score, mean, len(env_memory), meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if agent.epoch >= episodes:
            break

    env.close()
    agent.close()