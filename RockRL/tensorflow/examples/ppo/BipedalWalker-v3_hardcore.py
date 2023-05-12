import os
import gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras import layers

from RockRL.utils.vectorizedEnv import VectorizedEnv
from RockRL.utils.misc import MeanAverage
from RockRL.utils.memory import Memory
from RockRL.tensorflow import PPOAgent

def actor_model(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, activation='relu')(X_input)
    X = Dense(256, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    action = Dense(action_space, activation="tanh")(X)
    sigma = Dense(action_space, activation='softplus')(X)
    action_sigma = layers.concatenate([action, sigma])

    model = Model(inputs = X_input, outputs = action_sigma)
    return model

def critic_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(512, activation="relu")(X_input)
    X = Dense(256, activation="relu")(X)
    X = Dense(64, activation="relu")(X)
    value = Dense(1, activation=None)(X)

    model = Model(inputs = X_input, outputs = value)
    return model

from gym.wrappers import TimeLimit
if __name__ == "__main__":
    env_name = 'BipedalWalker-v3'

    num_envs = 48
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name, hardcore=True) # , render_mode="human")
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        # actor = actor_model(input_shape, action_space),
        # critic = critic_model(input_shape),
        actor = load_model("runs/1683805598/BipedalWalker-v3_actor.h5"), # , compile=False),
        critic = load_model("runs/1683805598/BipedalWalker-v3_critic.h5"), # , compile=False),
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        action_space="continuous",
        gamma=0.99,
        lamda=0.95,
        c2=0.001,
        batch_size=256
    )
    # agent.actor.load_weights("runs/1683795474/BipedalWalker-v3_actor.h5")
    # agent.critic.load_weights("runs/1683795474/BipedalWalker-v3_critic.h5")

    memory = Memory(num_envs=num_envs)
    meanAverage = MeanAverage(
        window_size=200,
        best_mean_score=-np.inf,
        best_mean_score_episode=500,
    )
    states = env.reset()
    episodes = 100000
    episode = 0
    while True:

        actions, probs = agent.act(states)

        next_states, rewards, dones = env.step(actions)
        memory.append(states, actions, rewards, probs, dones, next_states)
        states = next_states

        rewards_len = np.array([len(r) for r in memory.rewards])
        if np.any(rewards_len >= env._max_episode_steps):
            dones = np.array([True if r >= env._max_episode_steps else False for r in rewards_len])

        for index in np.where(dones)[0]:
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

            if not meanAverage.is_improoving(episode):
                agent.reduce_learning_rate(ratio=0.95, verbose=True, min_lr = 1e-07)

            print(episode, score, mean, len(_rewards), meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if episode >= episodes:
            break

    env.close()

    # env = gym.make(env_name) # , render_mode="human")
    # action_space = env.action_space.shape[0]
    # input_shape = env.observation_space.shape
    # low, high = env.action_space.low, env.action_space.high

    # agent = PPOAgent(
    #     actor = actor_model(input_shape, action_space),
    #     critic = critic_model(input_shape),
    #     actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #     critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #     action_space="continuous",
    #     batch_size=256
    # )

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
    #         if done or len(memory.rewards[0]) >= env._max_episode_steps: #  or score < -200:
                
    #             states, actions, rewards, predictions, dones, next_state = memory.get()
    #             agent.train(states, actions, rewards, predictions, dones, next_state)
                
    #             mean = meanAverage(score)
    #             print(episode, score, mean, len(memory.rewards[0]))
    #             break