import os
import gymnasium as gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    num_envs = 24
    env = VectorizedEnv(env_object=gym.make, num_envs=num_envs, id=env_name)
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

    memory = MemoryManager(num_envs=num_envs)
    meanAverage = MeanAverage()
    states, _ = env.reset()
    episodes = 10000
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