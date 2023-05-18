import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
# visible only one gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from keras.models import Model, load_model

if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    env = gym.make(env_name, render_mode="rgb_array")

    actor = load_model("runs/1684414014/LunarLander-v2_actor.h5", compile=False)
    actor.summary()

    states = env.reset()[0]
    episodes = 100
    score = 0
    episode = 0
    while True:
        a = env.render()
        probs = actor.predict(np.expand_dims(states, axis=0), verbose=False)
        actions = np.argmax(probs, axis=1)[0]

        states, rewards, dones = env.step(actions)[:3]
        score += rewards

        if dones:
            states = env.reset()[0]

            episode += 1
            print(episode, score)
            score = 0

        if episode >= episodes:
            break

    env.close()