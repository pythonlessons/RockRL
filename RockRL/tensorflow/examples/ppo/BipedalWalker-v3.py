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



def actor_model(input_shape, action_space, activation="tanh"):
    X_input = Input(input_shape)
    X = Dense(512, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
    X = Dense(256, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    X = Dense(64, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    output = Dense(action_space, activation=activation)(X)

    model = Model(inputs = X_input, outputs = output)
    return model

def critic_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
    X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    value = Dense(1, activation=None)(X)

    model = Model(inputs = X_input, outputs = value)
    return model

if __name__ == "__main__":
    env_name = 'BipedalWalker-v3'

    env = gym.make(env_name)
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape
    low, high = env.action_space.low, env.action_space.high

    agent = PPOAgent(
        actor = actor_model(input_shape, action_space, activation="tanh"),
        critic = critic_model(input_shape),
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        action_space="continuous",
        batch_size=256
    )

    episodes = 10000
    memory = Memory() 
    meanAverage = MeanAverage()
    for episode in range(episodes):
        # Instantiate or reset games memory
        state = env.reset()[0]
        done, score = False, 0
        memory.reset()
        while True:
            # Actor picks an action
            action, prob = agent.act(state)
            # Retrieve new state, reward, and whether the state is terminal
            next_state, reward, done, _ = env.step(action)[:4]
            # Memorize (state, action, reward) for training
            memory.append(state, action, reward, prob, done, next_state)
            # Update current state
            state = next_state
            score += reward
            if done: #  or score < -200:
                
                states, actions, rewards, predictions, dones, next_state = memory.get()
                agent.train(states, actions, rewards, predictions, dones, next_state)
                
                mean = meanAverage(score)
                print(episode, score, mean)
                break