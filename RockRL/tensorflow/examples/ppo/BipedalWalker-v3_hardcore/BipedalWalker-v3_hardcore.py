import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow logging
import gym
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # visible only one gpu
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.models import Model, load_model
from keras import layers

from RockRL.utils.vectorizedEnv import VectorizedEnv, CustomEnv
from RockRL.utils.misc import MeanAverage
from RockRL.utils.memory import Memory
from RockRL.tensorflow import PPOAgent


def actor_model(input_shape, action_space):
    X_input = layers.Input(input_shape)
    X = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(X_input)
    X = layers.LSTM(32)(X)
    X = layers.Dense(32, kernel_initializer="he_uniform")(X)
    X = layers.LeakyReLU(alpha=0.1)(X)

    action = layers.Dense(action_space, activation="tanh")(X)
    sigma = layers.Dense(action_space, activation='softplus')(X)
    action_sigma = layers.concatenate([action, sigma])

    model = Model(inputs = X_input, outputs = action_sigma)
    return model

def critic_model(input_shape):
    X_input = layers.Input(input_shape)
    X = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(X_input)
    X = layers.LSTM(32)(X)
    X = layers.Dense(32, kernel_initializer="he_uniform")(X)
    X = layers.LeakyReLU(alpha=0.1)(X)

    value = layers.Dense(1, activation=None)(X)

    model = Model(inputs = X_input, outputs = value)
    return model


if __name__ == "__main__":
    env_name = 'BipedalWalker-v3'

    num_envs = 48
    env = VectorizedEnv(env_object=CustomEnv, custom_env_object=gym.make, os_hist_steps=4, num_envs=num_envs, id=env_name, hardcore=True)# , render_mode="human")
    action_space = env.action_space.shape[0]
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        actor = load_model("runs/1684395972/BipedalWalker-v3_actor.h5"), # load pretrained model on simple environment with optimizer
        critic = load_model("runs/1684395972/BipedalWalker-v3_critic.h5"),
        # actor = actor_model(input_shape, action_space),
        # critic = critic_model(input_shape),
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        action_space="continuous",
        batch_size=256,
        train_epochs=10,
        gamma=0.98,
        lamda=0.90,
        c2=0.001,
        compile=True,
    )
    agent.actor.summary()

    # agent.actor.load_weights("runs/1684395972/BipedalWalker-v3_actor.h5")
    # agent.critic.load_weights("runs/1684395972/BipedalWalker-v3_critic.h5")

    memory = Memory(num_envs=num_envs, input_shape=input_shape)
    meanAverage = MeanAverage(
        window_size=100,
        best_mean_score=-np.inf,
        best_mean_score_episode=2000,
    )
    states = env.reset()
    reduce_lr_episode = 2000
    wait_best_window = 100
    episodes = 100000
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
                agent.reduce_learning_rate(ratio=0.99, verbose=True, min_lr = 1e-05)
                reduce_lr_episode = episode + wait_best_window

            print(episode, score, mean, len(_rewards), meanAverage.best_mean_score, meanAverage.best_mean_score_episode)

        if episode >= episodes:
            break

    env.close()