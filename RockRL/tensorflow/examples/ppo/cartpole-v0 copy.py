import numpy as np
import tensorflow as tf 
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import gym
# import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

from keras import backend as K

# env = gym.make("CartPole-v0")
# low = env.observation_space.low
# high = env.observation_space.high

def actor(state_size, action_size):
    inputs = tf.keras.layers.Input(shape=(state_size,))
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    a = tf.keras.layers.Dense(action_size, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=a)

def critic(state_size):
    inputs = tf.keras.layers.Input(shape=(state_size,))
    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    v = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs=inputs, outputs=v)

# class critic(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.d1 = tf.keras.layers.Dense(128,activation='relu')
#         self.v = tf.keras.layers.Dense(1, activation = None)

#     def call(self, input_data):
#         x = self.d1(input_data)
#         v = self.v(x)
#         return v
    

# class actor(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.d1 = tf.keras.layers.Dense(128,activation='relu')
#         self.a = tf.keras.layers.Dense(2,activation='softmax')

#     def call(self, input_data):
#         x = self.d1(input_data)
#         a = self.a(x)
#         return a

def test_reward(env, actor):
    total_reward = 0
    state, info = env.reset()
    done = False
    while not done:
        # action = np.argmax(actor.predict(np.expand_dims(state, axis=0), verbose=False)[0])
        action = np.argmax(actor(np.expand_dims(state, axis=0), training=False).numpy()[0])
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        total_reward += reward
        if total_reward < -250:
            break

    return total_reward

# def compute_gaes(rewards, dones, values, next_values, gamma = 1.0, lmbda = 0.95, normalize=True):
#     g = 0
#     returns = []
#     for i in reversed(range(len(rewards))):
#        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
#        g = delta + gamma * lmbda * (1 - dones[i]) * g
#        returns.append(g + values[i])

#     returns.reverse()
#     adv = np.array(returns, dtype=np.float32) - values
#     if normalize:
#         adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

#     returns = np.array(returns, dtype=np.float32)
#     return returns, adv    

class PPOAgent(tf.keras.models.Model):
    def __init__(
            self,
            actor,
            critic,
            clip_pram=0.2,
            c1=0.5,
            c2=0.0001,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=True,
        ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.clip_pram = clip_pram
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def compile(
            self, 
            a_opt=tf.keras.optimizers.Adam(learning_rate=0.001),
            c_opt=tf.keras.optimizers.Adam(learning_rate=0.001),
            **kwargs
        ):
        super().compile(**kwargs)
        self.a_opt = a_opt
        self.c_opt = c_opt

    def ciritc_loss(self, y_true, y_pred):
        critic_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return critic_loss
  
    def actor_loss(self, probs, actions, advantages, old_probs):

        actions_oh = K.one_hot(actions, probs.shape[-1])

        # # get best onehot actions from probs
        # probs_oh = K.one_hot(K.argmax(probs), probs.shape[-1])

        action_probs = K.sum(probs * actions_oh, axis=1)
        old_action_probs = K.sum(old_probs * actions_oh, axis=1)

        # Entropy regularization
        # entropy = K.mean(-action_probs * K.log(action_probs + 1e-10)) * self.c2

        # Probability calculation
        log_p = K.log(action_probs)
        old_log_p = K.log(old_action_probs)

        # Surrogate objective
        ratio = K.exp(log_p - old_log_p)
        clipped_ratio = K.clip(ratio, 1 - self.clip_pram, 1 + self.clip_pram)
        sr1 = ratio * advantages
        sr2 = clipped_ratio * advantages
        # surrogate = tf.where(ratio > clipped_ratio, sr2, sr1)
        actor_loss = -K.mean(K.minimum(sr1, sr2))

        # Entropy loss
        entropy = -K.mean(probs * K.log(probs + 1e-10)) * self.c2

        # Final loss
        a_loss = actor_loss # - entropy
        # a_loss = -tf.reduce_mean(surrogate) - tf.reduce_mean(entropy)
        return a_loss, entropy
  
    
    def compute_gaes(self, rewards, dones, values, next_values, gamma=1.0, lamda=0.95, normalize=True):
        """ Compute Generalized Advantage Estimation using NumPy."""
        values = np.array(values, dtype=np.float32)
        next_values = np.array(next_values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = 1 - np.array(dones, dtype=np.float32)
        deltas = rewards + gamma * next_values * dones - values
        gaes = np.copy(deltas)

        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + dones[t] * gamma * lamda * gaes[t + 1]

        returns = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return returns, gaes

    def train_step(self, data):
        states, actions, advantages, old_probs, discounted_rewards = data[0]

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probs = self.actor(states, training=True)
            values = self.critic(states, training=True)
            c_loss = self.c1 * self.ciritc_loss(discounted_rewards, values)
            # c_loss = tf.reduce_mean(tf.square(values - discounted_rewards)) * self.c1
            # c_loss = kls.mean_squared_error(discounted_rewards, values) * self.c1
            # c_loss = self.c1 * self.ciritc_loss(next_values, values)
            a_loss, entropy = self.actor_loss(probs, actions, advantages, old_probs)
            total_loss = a_loss - entropy# - c_loss

        # Compute gradients and update networks
        # grads = tape1.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        # self.a_opt.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        grads1 = tape1.gradient(total_loss, self.actor.trainable_variables)
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))

        results = {"a_loss": total_loss, "c_loss": c_loss, "entropy": entropy}

        return results
    
    def act(self, state):
        prob = self.actor(np.expand_dims(state, axis=0), training=False).numpy()[0]
        # prob = self.actor.predict(np.expand_dims(state, axis=0), verbose=False)[0]
        action = np.random.choice(prob.shape[0], p=prob)
        return action, prob


# env = gym.make("CartPole-v0", max_episode_steps=500)
env = gym.make("LunarLander-v2", max_episode_steps=1000)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

ppo_agent = PPOAgent(
    actor(state_size, action_size), 
    critic(state_size)
    )

ppo_agent.compile(
    a_opt=tf.keras.optimizers.Adam(learning_rate=0.00025),
    c_opt=tf.keras.optimizers.Adam(learning_rate=0.00025),
    run_eagerly=False
    )

steps = 5000
ep_reward = []
total_avgr = []
best_reward = 0 # -np.inf
avg_rewards_list = []

for s in range(steps):
  
    done = False
    state, info = env.reset()
    states, actions, rewards, probs, dones, next_states = [], [], [], [], [], []
    print("new episode")

    for e in range(2048):
    # while True:
   
        action, prob = ppo_agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        dones.append(done)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        actions.append(action)
        probs.append(prob)
        state = next_state
        if done: #  or np.sum(rewards)<-250:
            env.reset()
            # break
  
        # if len(states) >= 500:
        #     break

    critic_preds = ppo_agent.critic(np.array(states + [next_state]), training=False).numpy().squeeze()
    # critic_preds = ppo_agent.critic.predict(np.array(states + [next_state]), verbose = False).squeeze()
    values, next_values = critic_preds[:-1], critic_preds[1:]
    probs = np.array(probs)
    states = np.array(states)
    actions = np.array(actions)

    # returns, advantages = ppo_agent.compute_gaes(rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True)
    returns, advantages = ppo_agent.compute_gaes(rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True)

    ppo_agent.fit(x=(states, actions, advantages, probs, returns), epochs=10, verbose=False, batch_size=256)   
    
    # if s > 100:
    avg_reward = np.mean([test_reward(env, ppo_agent.actor) for _ in range(2)])
    print(f"{s} total test reward is {avg_reward}")
    avg_rewards_list.append(avg_reward)
    if avg_reward >= best_reward:
        print('best reward=' + str(avg_reward))
        best_reward = avg_reward
        # change learning rate of actor and critic
        ppo_agent.a_opt.lr.assign(ppo_agent.a_opt.lr.numpy() * 0.95)
        ppo_agent.c_opt.lr.assign(ppo_agent.c_opt.lr.numpy() * 0.95)
        print(f"lr: {ppo_agent.a_opt.lr.numpy()}")
    # if best_reward == 200:
    #     target = True
    env.reset()

env.close()
    