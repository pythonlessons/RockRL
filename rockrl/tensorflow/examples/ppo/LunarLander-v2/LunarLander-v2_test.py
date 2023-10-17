import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gymnasium as gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

from keras import models

if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    # env = gym.make(env_name)
    env = gym.make(env_name, render_mode="human")

    actor = models.load_model("runs/1696937445/LunarLander-v2_actor.h5", compile=False)
    actor.summary()

    state, info = env.reset()
    episodes, episode, score = 100, 0, 0
    while True:
        probs = actor.predict(np.expand_dims(state, axis=0), verbose=False)
        action = np.argmax(probs, axis=1)[0]

        state, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()

        if terminated or truncated:
            state, info = env.reset()

            episode += 1
            print(episode, score)
            score = 0

        if episode >= episodes:
            break

    env.close()