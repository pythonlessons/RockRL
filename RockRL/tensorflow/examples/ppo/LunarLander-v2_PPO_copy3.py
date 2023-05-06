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

def actor_model(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, activation='relu')(X_input)
    # X = LeakyReLU(alpha=0.1)(X)
    X = Dense(256, activation='relu')(X)
    # X = LeakyReLU(alpha=0.1)(X)
    X = Dense(64, activation='relu')(X)
    # X = LeakyReLU(alpha=0.1)(X)
    output = Dense(action_space, activation="softmax")(X)

    model = Model(inputs = X_input, outputs = output)
    return model

def critic_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(512, activation='relu')(X_input)
    # X = LeakyReLU(alpha=0.1)(X)
    X = Dense(256, activation='relu')(X)
    # X = LeakyReLU(alpha=0.1)(X)
    X = Dense(64, activation='relu')(X)
    # X = LeakyReLU(alpha=0.1)(X)
    value = Dense(1, activation=None)(X)

    model = Model(inputs = X_input, outputs = value)
    return model


class PPOAgent(tf.keras.models.Model):
    def __init__(
        self, 
        actor,
        critic,
        loss_clipping=0.2,
        c1=0.5,
        c2=0.01,
        ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.loss_clipping = loss_clipping # epsilon in clipped loss
        self.c1 = c1 # value coefficient
        self.c2 = c2 # entropy coefficient

    def compile(
        self, 
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
        **kwargs
        ):
        super().compile(**kwargs)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def entropy(self, y_pred):
        return -K.mean(y_pred * K.log(y_pred + 1e-10), axis=1) * self.c2

    def critic_loss(self, y_pred, target):
        loss = self.c1 * K.mean((target - y_pred) ** 2)
        return loss
    
    def actor_loss(self, y_pred, advantages, predictions, actions):
        # Defined in https://arxiv.org/abs/1707.06347

        # testing to replace prob
        # argmax = K.argmax(y_pred, axis=1)
        # oh_actions = tf.one_hot(argmax, y_pred.shape[-1])
        # prob = oh_actions * y_pred

        actions_onehot = K.one_hot(actions, y_pred.shape[-1])

        prob = actions_onehot * y_pred
        old_prob = actions_onehot * predictions

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value = 1-self.loss_clipping, max_value = 1+self.loss_clipping) * advantages

        loss = -K.mean(K.minimum(p1, p2))

        return loss

        entropy = self.c2 * K.mean(-(y_pred * K.log(y_pred + 1e-10)))
        
        total_loss = loss - entropy

        return total_loss
    
    def act(self, state):
        state_dim = state.ndim
        if state_dim == 1:
            state = np.expand_dims(state, axis=0)

        # Use the network to predict the next action to take, using the model
        probs = self.actor(state, training=False).numpy()
        actions = np.array([np.random.choice(prob.shape[0], p=prob) for prob in probs])
    
        if state_dim == 1:
            return actions[0], probs[0]

        return actions, probs
    
    @tf.function
    def get_gaes_tf(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        rewards = tf.cast(rewards, tf.float32)
        dones = 1 - tf.cast(dones, tf.float32)
        gaes = rewards + gamma * next_values * dones - values 

        start = gaes.shape[0] - 2
        indices = tf.range(start, -1, -1)

        for t in indices:
            update = dones[t] * gamma * lamda * gaes[t + 1]
            gaes = tf.tensor_scatter_nd_add(gaes, [[t]], [update])

        target = gaes + values
        if normalize and gaes.shape[0] > 1:
            gaes = (gaes - K.mean(gaes)) / (K.std(gaes) + 1e-8)
        
        return tf.expand_dims(gaes, axis=1), tf.expand_dims(target, axis=1)

    # TODO use this for training
    def predict_chunks(self, model, data, batch_size=64, training=True):
        predictions = []
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            predictions.append(model(batch, training=training))

        return tf.concat(predictions, axis=0)

    @tf.function
    def train_step(self, data):
        states, predictions, actions, rewards, dones, next_state = data # [0]

        # numpy to tensor
        # states = tf.convert_to_tensor(states, dtype=tf.float32)
        # next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        next_state = tf.expand_dims(next_state, axis=0)
        combined_states = tf.concat([states, next_state], axis=0)
        # next_states = combined_states[1:]

        values_pred = self.predict_chunks(self.critic, combined_states, batch_size=64, training=False)

        values = tf.squeeze(values_pred[:-1])
        next_values = tf.squeeze(values_pred[1:])

        advantages, target = self.get_gaes_tf(rewards, dones, values, next_values)

        # advantages, target = self.get_gaes_tf(rewards, dones, values, next_values)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # probs = self.actor(states, training=True)  # Forward pass
            probs = self.predict_chunks(self.actor, states, batch_size=64, training=True)  # Forward pass

            # values_pred = self.critic(states, training=True)
            # values_pred = self.critic(combined_states, training=True)

            values = self.predict_chunks(self.critic, states, batch_size=64, training=True)
            # # values_pred = self.critic(combined_states, training=True)

            # values = tf.squeeze(values_pred[:-1])
            # next_values = tf.squeeze(values_pred[1:])

            # advantages, target = self.get_gaes_tf(rewards, dones, values, next_values)

            # Compute the loss value
            actor_loss = self.actor_loss(probs, advantages, predictions, actions)
            critic_loss = self.critic_loss(values, target)
            entropy = self.entropy(probs)

            total_loss = actor_loss - entropy

        # Compute gradients
        grads_actor = tape1.gradient(total_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

        # Compute gradients
        grads_critic = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        return {"a_loss": actor_loss, "c_loss": critic_loss}

    def train(self, states, actions, rewards, predictions, dones, next_state):
        self.epochs = 1
        # self.shuffle = False
        # reshape memory to appropriate shape for training
        predictions = np.vstack(predictions)

        # # Get Critic network predictions 
        # # append next state to states
        # all_states = np.array(states + [next_state])
        # # next_states = np.array(all_states[1:])
        # all_values = self.critic(all_states, training=False).numpy().squeeze()
        # values, next_values = all_values[:-1], all_values[1:]

        # # Compute discounted rewards and advantages
        # advantages, target = self.get_gaes(rewards, dones, values, next_values)
        # # advantages, target = self.get_gaes_tf(rewards, dones, values, next_values)

        states = np.array(states)
        actions = np.array(actions)

        rewards = np.array(rewards)
        dones = np.array(dones)

        # self.train_step((states, advantages, predictions, actions, target, rewards, dones, values, next_values, next_state))
        for _ in range(self.epochs):
            self.train_step((states, predictions, actions, rewards, dones, next_state))

        # self.fit(x=(states, advantages, predictions, actions, target, rewards, dones, values, next_values), epochs=self.epochs, shuffle=False, verbose=False)

# import ThreadPoolExecutor and ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def run_env(conn, env_name):
    env = gym.make(env_name)

    while True:
        # Wait for a message on the connection
        message = conn[1].recv()

        if isinstance(message, str):
            if message == 'reset':
                state = env.reset()[0]
                conn[1].send(state)
            elif message == 'close':
                env.close()
                break

        else:
            # Assume it's a tuple of (action, render)
            action = message
            state, reward, done = env.step(action)[:3]
            conn[1].send((state, reward, done))

class VectorizedEnv:
    def __init__(self, env_name: str, num_envs: int=2, **kwargs):
        self.env_name = env_name

        self.conns = [mp.Pipe() for _ in range(num_envs)]
        self.envs = [mp.Process(target=run_env, args=(conn, env_name,)) for conn in self.conns]
        # self.envs = []
        # self.conns = []
        # for _ in range(num_envs):
        #     conn = mp.Pipe()
        #     env = mp.Process(target=self.run_env, args=(conn,))
        #     env.start()
        #     self.envs.append(env)
        #     self.conns.append(conn)
        for env in self.envs:
            env.start()
        # self.envs = [gym.make(env_name, **kwargs) for _ in range(num_envs)]
        # self.num_envs = num_envs
        # self.input_shape = self.envs[0].observation_space.shape
        # self.action_space = self.envs[0].action_space.n
        # self.executor = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="EnvWorker")

    def reset(self, index=None): # return state
        if index is None:
            for conn in self.conns:
                conn[0].send('reset')
            states = np.array([conn[0].recv() for conn in self.conns])
        else:
            self.conns[index][0].send('reset')
            states = self.conns[index][0].recv()

        return states
        #     state = np.array([env.reset()[0] for env in self.envs])
        #     return state
        # else:
        #     return self.envs[index].reset()[0]
        
    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn[0].send(action)

        results = [conn[0].recv() for conn in self.conns]
        # futures = [self.executor.submit(self.envs[i].step, actions[i]) for i in range(self.num_envs)]
        # results = [future.result()[:3] for future in futures]
        next_states, rewards, dones = zip(*results)
        
        return np.array(next_states), np.array(rewards), np.array(dones)
    
    def close(self):
        pass
        # for env in self.envs:
        #     env.close()

class Memory:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.reset()

    def reset(self, index=None):
        if index is None:
            self.states = [[] for _ in range(self.num_envs)]
            self.actions = [[] for _ in range(self.num_envs)]
            self.rewards = [[] for _ in range(self.num_envs)]
            self.predictions = [[] for _ in range(self.num_envs)]
            self.dones = [[] for _ in range(self.num_envs)]
            self.next_states = [None for _ in range(self.num_envs)]

        else:
            self.states[index] = []
            self.actions[index] = []
            self.rewards[index] = []
            self.predictions[index] = []
            self.dones[index] = []
            self.next_states[index] = None

    def append(self, states, actions, rewards, predictions, dones, next_states):
        # check whether the input is dimensioned for one or multiple environments
        if states.ndim == 1:
            states, actions, rewards, predictions, dones, next_states = [[x] for x in [states, actions, rewards, predictions, dones, next_states]]

        for i in range(self.num_envs):
            self.states[i].append(states[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.predictions[i].append(predictions[i])
            self.dones[i].append(dones[i])
            self.next_states[i] = next_states[i]

    def get(self, index=None):
        if self.num_envs == 1:
            return self.states[0], self.actions[0], self.rewards[0], self.predictions[0], self.dones[0], self.next_states[0]
        
        if index is None:
            raise ValueError("Must provide indexes when using multiple environments")
        return self.states[index], self.actions[index], self.rewards[index], self.predictions[index], self.dones[index], self.next_states[index]

class MeanAverage:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.values = []

    def __call__(self, x):
        self.values.append(x)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        return np.mean(self.values)

if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    num_envs = 8
    env = VectorizedEnv(env_name, num_envs=num_envs)
    action_space = 4 # env.action_space
    input_shape = (8,) # env.input_shape

    agent = PPOAgent(
        actor = actor_model(input_shape, action_space),
        critic = critic_model(input_shape)
    )
    agent.compile(run_eagerly=False)

    memory = Memory(num_envs=num_envs)
    meanAverage = MeanAverage()
    states = env.reset()
    episodes = 10000
    episode = 0
    while True:
        actions, probs = agent.act(states)

        next_states, rewards, dones = env.step(actions)
        memory.append(states, actions, rewards, probs, dones, next_states)
        states = next_states

        rewards_sum = np.array([np.sum(r) for r in memory.rewards])
        if np.any(rewards_sum < -350):
            dones = np.array([True if r < -350 else False for r in rewards_sum])

        for index in np.where(dones)[0]:
            _states, _actions, _rewards, _predictions, _dones, _next_state = memory.get(index=index)
            agent.train(_states, _actions, _rewards, _predictions, _dones, _next_state)
            memory.reset(index=index)
            states[index] = env.reset(index)

            episode += 1
            score = np.sum(_rewards)
            mean = meanAverage(score)
            print(episode, score, mean)

        if episode >= episodes:
            break
                


    # env = gym.make(env_name)
    # action_space = env.action_space.n
    # input_shape = env.observation_space.shape

    # agent = PPOAgent(
    #     actor = actor_model(input_shape, action_space),
    #     critic = critic_model(input_shape)
    # )
    # agent.compile(run_eagerly=False)

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