import os
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gym
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.models import Model, load_model
from keras.layers import Input, Dense, LeakyReLU
from keras import backend as K

from RockRL.utils.misc import MeanAverage
from RockRL.utils.memory import Memory
from RockRL.tensorflow.PPO.ppo import PPOAgent

from concurrent.futures import ThreadPoolExecutor

def actor_model(input_shape, action_space):
    X_input = Input(input_shape)
    # X = Dense(512, activation='relu')(X_input)
    # # X = LeakyReLU(alpha=0.1)(X)
    # X = Dense(256, activation='relu')(X)
    # # X = LeakyReLU(alpha=0.1)(X)
    # X = Dense(64, activation='relu')(X)
    # # X = LeakyReLU(alpha=0.1)(X)
    X = Dense(512)(X_input)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(256)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(64)(X)
    X = LeakyReLU(alpha=0.1)(X)
    # X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
    # X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    # X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    output = Dense(action_space, activation="softmax")(X)

    model = Model(inputs = X_input, outputs = output)
    return model

def critic_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(512)(X_input)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(256)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(64)(X)
    X = LeakyReLU(alpha=0.1)(X)
    # X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
    # X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    # X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    value = Dense(1, activation=None)(X)

    model = Model(inputs = X_input, outputs = value)
    return model

class VectorizedEnv:
    def __init__(self, env_name, num_envs=2, **kwargs):
        self.envs = [gym.make(env_name, **kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.input_shape = self.envs[0].observation_space.shape
        self.action_space = self.envs[0].action_space.n
        self.executor = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="EnvWorker")

    def reset(self, index=None):
        if index is None:
            state = np.array([env.reset()[0] for env in self.envs])
            return state
        else:
            return self.envs[index].reset()[0]
        
    def step(self, actions):
        futures = [self.executor.submit(self.envs[i].step, actions[i]) for i in range(self.num_envs)]
        results = [future.result()[:3] for future in futures]
        next_states, rewards, dones = zip(*results)
        
        return np.array(next_states), np.array(rewards), np.array(dones)


if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    # num_envs = 8
    # env = VectorizedEnv(env_name, num_envs=num_envs)
    # action_space = env.action_space
    # input_shape = env.input_shape

    # agent = PPOAgent(
    #     actor = actor_model(input_shape, action_space),
    #     critic = critic_model(input_shape)
    # )
    # agent.compile(
    #     actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    #     critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    #     run_eagerly=False
    #     )

    # memory = Memory(num_envs=num_envs)
    # meanAverage = MeanAverage()
    # states = env.reset()
    # episodes = 10000
    # episode = 0
    # while True:
    #     actions, probs = agent.act(states)

    #     next_states, rewards, dones = env.step(actions)
    #     memory.append(states, actions, rewards, probs, dones, next_states)
    #     states = next_states

    #     rewards_sum = np.array([np.sum(r) for r in memory.rewards])
    #     if np.any(rewards_sum < -500):
    #         dones = np.array([True if r < -500 else False for r in rewards_sum])

    #     for index in np.where(dones)[0]:
    #         _states, _actions, _rewards, _predictions, _dones, _next_state = memory.get(index=index)
    #         agent.train(_states, _actions, _rewards, _predictions, _dones, _next_state)
    #         memory.reset(index=index)
    #         states[index] = env.reset(index)

    #         episode += 1
    #         score = np.sum(_rewards)
    #         mean = meanAverage(score)
    #         print(episode, score, mean)

    #     if episode >= episodes:
    #         break
                


    env = gym.make(env_name)
    action_space = env.action_space.n
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        actor = actor_model(input_shape, action_space),
        critic = critic_model(input_shape)
    )
    agent.compile(
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        run_eagerly=False
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
            if done or score < -350:
                
                states, actions, rewards, predictions, dones, next_state = memory.get()
                agent.train(states, actions, rewards, predictions, dones, next_state)
                
                mean = meanAverage(score)
                print(episode, score, mean)
                break