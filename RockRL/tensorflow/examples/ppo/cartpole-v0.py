import numpy as np
import tensorflow as tf 
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

from keras import backend as K

env= gym.make("CartPole-v0")
low = env.observation_space.low
high = env.observation_space.high

class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    v = self.v(x)
    return v
    

class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.a = tf.keras.layers.Dense(2,activation='softmax')

  def call(self, input_data):
    x = self.d1(input_data)
    a = self.a(x)
    return a
     
class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2

          
    def act(self, state):
        prob = self.actor(np.expand_dims(state, axis=0), training=False).numpy()[0]
        # prob = self.actor.predict(np.expand_dims(state, axis=0), verbose=False)[0]
        action = np.random.choice(prob.shape[0], p=prob)
        return action, prob
        # dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        # action = dist.sample()
        # return int(action.numpy()[0]), prob[0]
    
    def ciritc_loss(self, y_true, y_pred):
        critic_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return critic_loss
  
    def actor_loss(self, probs, actions, adv, old_probs):
        # Entropy regularization
        entropy = tf.reduce_mean(-(probs * tf.math.log(probs + 1e-10)), axis=-1) * 0.001

        # Probability calculation
        actions_oh = tf.one_hot(actions, probs.shape[-1])
        log_p = tf.math.log(tf.reduce_sum(probs * actions_oh, axis=-1))
        old_log_p = tf.math.log(tf.reduce_sum(old_probs * actions_oh, axis=-1))

        # Surrogate objective
        ratio = tf.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_pram, 1 + self.clip_pram)
        sr1 = ratio * adv
        sr2 = clipped_ratio * adv
        surrogate = tf.where(ratio > clipped_ratio, sr2, sr1)

        # Final loss
        a_loss = -tf.reduce_mean(surrogate) - tf.reduce_mean(entropy)
        return a_loss
    

    def learn(self, states, actions, adv, old_probs, discnt_rewards):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probs = self.actor(states, training=True)
            values = self.critic(states, training=True)
            c_loss = 0.5 * self.ciritc_loss(discnt_rewards, values)
            a_loss = self.actor_loss(probs, actions, adv, old_probs)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))

        return a_loss, c_loss
    
def test_reward(env):
    total_reward = 0
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(agentoo7.actor.predict(np.expand_dims(state, axis=0), verbose=False)[0])
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward

def compute_gaes(rewards, dones, values, next_values, gamma = 1.0, lmbda = 0.95, normalize=True):
    g = 0
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
       g = delta + gamma * lmbda * (1 - dones[i]) * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values
    if normalize:
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    returns = np.array(returns, dtype=np.float32)
    return returns, adv    



tf.random.set_seed(336699)
agentoo7 = agent()
steps = 5000
ep_reward = []
total_avgr = []
target = False 
best_reward = 0
avg_rewards_list = []


for s in range(steps):
    if target == True:
        break
  
    done = False
    state, info = env.reset()
    rewards = []
    states = []
    actions = []
    probs = []
    dones = []
    next_states = []
    print("new episode")

    for e in range(128):
    # while True:
   
        action, prob = agentoo7.act(state)
        next_state, reward, done, _, info = env.step(action)
        dones.append(done)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        actions.append(action)
        probs.append(prob)
        state = next_state
        if done:
            env.reset()
            # break
  
    critic_preds = agentoo7.critic(np.array(states + [next_state]), training=False).numpy().squeeze()
    # critic_preds = agentoo7.critic.predict(np.array(states + [next_state]), verbose = False).squeeze()
    values, next_values = critic_preds[:-1], critic_preds[1:]
    probs = np.array(probs)
    states = np.array(states)
    actions = np.array(actions)

    returns, adv = compute_gaes(rewards, dones, values, next_values)

    for epocs in range(1):
        al, cl = agentoo7.learn(states, actions, adv, probs, returns)  

    avg_reward = np.mean([test_reward(env) for _ in range(5)])
    print(f"total test reward is {avg_reward}")
    avg_rewards_list.append(avg_reward)
    if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            best_reward = avg_reward
    if best_reward == 200:
        target = True
    env.reset()

env.close()
    