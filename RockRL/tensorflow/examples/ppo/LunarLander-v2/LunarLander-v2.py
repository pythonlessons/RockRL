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

    num_envs = 48
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

        done_indices = np.where(dones)[0]
        rewards_len = np.array([len(r) for r in memory.rewards])
        max_episodes = np.where(rewards_len >= env._max_episode_steps)[0]
        if max_episodes.any():
            done_indices = np.unique(np.concatenate((done_indices, max_episodes)))

        for index in done_indices:
            _states, _actions, _rewards, _predictions, _dones, _next_state = memory.get(index=index)
            agent.train(_states, _actions, _rewards, _predictions, _dones, _next_state)
            memory.reset(index=index)
            states[index] = env.reset(index)

            episode += 1
            score = np.sum(_rewards)
            mean = meanAverage(score)

            if meanAverage.is_best(episode):
                # save model
                pass

            if meanAverage.is_best(episode):
                # save model
                agent.save_models(env_name)

                if meanAverage.best_mean_score_episode > reduce_lr_episode:
                    reduce_lr_episode = meanAverage.best_mean_score_episode + wait_best_window

            if reduce_lr_episode < episode:
                agent.reduce_learning_rate(ratio=0.99, verbose=True, min_lr = 5e-06)
                reduce_lr_episode = episode + wait_best_window

            print(episode, score, mean, meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if episode >= episodes:
            break

    env.close()
                


    # env = gym.make(env_name)
    # action_space = env.action_space.n
    # input_shape = env.observation_space.shape

    # agent = PPOAgent(
    #     actor = actor_model(input_shape, action_space),
    #     critic = critic_model(input_shape),
    #     batch_size=256
    # )
    # agent.compile(
    #     actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    #     critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    #     run_eagerly=False
    #     )

    # episodes = 10000
    # memory = Memory() 
    # meanAverage = MeanAverage()
    # for episode in range(episodes):
    #     # Instantiate or reset games memory
    #     state = env.reset()[0]
    #     done, score = False, 0
    #     memory.reset()
    #     while True:
    #         # Actor picks an action
    #         action, prob = agent.act(state)
    #         # Retrieve new state, reward, and whether the state is terminal
    #         next_state, reward, done, _ = env.step(action)[:4]
    #         # Memorize (state, action, reward) for training
    #         memory.append(state, action, reward, prob, done, next_state)
    #         # Update current state
    #         state = next_state
    #         score += reward
    #         if done or score < -350:
                
    #             states, actions, rewards, predictions, dones, next_state = memory.get()
    #             agent.train(states, actions, rewards, predictions, dones, next_state)
                
    #             mean = meanAverage(score)
    #             print(episode, score, mean)
    #             break