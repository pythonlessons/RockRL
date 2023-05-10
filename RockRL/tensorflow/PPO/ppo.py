import os
import time
import numpy as np
import tensorflow as tf
from keras import backend as K


class PPOAgent:
    def __init__(
        self, 
        actor: tf.keras.models.Model,
        critic: tf.keras.models.Model,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        action_space: str="discrete", # "discrete" or "continuous"
        loss_clipping: float=0.2,
        c1: float=0.5,
        c2: float=0.001,
        train_epochs: int=10,
        batch_size: int=64,
        epoch: int = 0,
        logdir: str = f"runs/{int(time.time())}", # use timestamp as default logdir
        writer = None
        ):
        # super().__init__()
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.action_space = action_space
        self.loss_clipping = loss_clipping # epsilon in clipped loss
        self.c1 = c1 # value coefficient
        self.c2 = c2 # entropy coefficient
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.epoch = epoch

        # Tensorboard logging
        self.logdir = logdir
        self.writer = writer

        if self.logdir and not writer:
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = tf.summary.create_file_writer(self.logdir)

    # def compile(
    #     self, 
    #     actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #     critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #     **kwargs
    #     ):
    #     super().compile(**kwargs)
    #     self.actor_optimizer = actor_optimizer
    #     self.critic_optimizer = critic_optimizer

    def entropy(self, probs):
        return -K.mean(probs * K.log(probs + 1e-10)) * self.c2

    def critic_loss(self, y_pred, target):
        y_pred = tf.squeeze(y_pred)
        target = tf.squeeze(target)
        loss = self.c1 * K.mean((y_pred - target) ** 2)
        return loss
    
    def logprob_dist(self, probs, actions):
        oh_actions = K.one_hot(tf.cast(actions, tf.uint8), probs.shape[-1])
        probs = K.sum(oh_actions * probs, axis=1)
        log_probs = K.log(probs + 1e-10)

        return log_probs
    
    def actor_loss(self, probs, advantages, old_probs, actions):
        # Defined in https://arxiv.org/abs/1707.06347

        if self.action_space == "discrete":
            log_prob = self.logprob_dist(probs, actions)
            log_old_prob = self.logprob_dist(old_probs, actions)
        else:
            log_prob = self.gaussian_likelihood(probs, actions)
            log_old_prob = self.gaussian_likelihood(old_probs, actions)

        log_ratio = log_prob - log_old_prob
        ratio = K.exp(log_ratio)
        
        p1 = advantages * ratio
        p2 = advantages * K.clip(ratio, 1-self.loss_clipping, 1+self.loss_clipping)

        loss = -K.mean(K.minimum(p1, p2))

        # approx_kl_divergence = K.mean((K.exp(log_ratio) - 1) - log_ratio)

        return loss# , approx_kl_divergence
    
    def gaussian_likelihood(self, probs, actions):
        actions = tf.cast(actions, tf.float32)
        log_std = -0.5 * K.ones(probs.shape[-1], dtype=np.float32)
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        # pre_sum = -0.5 * (((actions-probs)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi)) 
        pre_sum = -0.5 * (((actions-probs)/(K.exp(log_std)+1e-10))**2 + 2*log_std + K.log(6.283185307179586))
        return K.sum(pre_sum, axis=1)


    def act(self, state):
        state_dim = state.ndim
        if state_dim == 1:
            state = np.expand_dims(state, axis=0)

        # Use the network to predict the next action to take, using the model
        probs = self.actor(state, training=False).numpy()
        if self.action_space == "discrete":
            actions = np.array([np.random.choice(prob.shape[0], p=prob) for prob in probs])
        else:
            _log_std = -0.5 * np.ones(probs.shape, dtype=np.float32)
            _std = np.exp(_log_std)

            low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
            actions = probs + np.random.uniform(low, high, size=probs.shape) * _std
            actions = np.clip(actions, low, high)

            # probs = self.gaussian_likelihood(actions, probs)
    
        if state_dim == 1:
            return actions[0], probs[0]

        return actions, probs
    
    # @tf.function
    # def get_gaes_tf(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
    #     rewards = tf.cast(rewards, tf.float32)
    #     dones = 1 - tf.cast(dones, tf.float32)
    #     gaes = rewards + gamma * next_values * dones - values 

    #     start = gaes.shape[0] - 2
    #     indices = tf.range(start, -1, -1)

    #     for t in indices:
    #         update = dones[t] * gamma * lamda * gaes[t + 1]
    #         gaes = tf.tensor_scatter_nd_add(gaes, [[t]], [update])

    #     target = gaes + values
    #     if normalize and gaes.shape[0] > 1:
    #         gaes = (gaes - K.mean(gaes)) / (K.std(gaes) + 1e-8)
        
    #     return gaes, target

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        values = np.array(values, dtype=np.float32)
        next_values = np.array(next_values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = 1 - np.array(dones, dtype=np.float32)
        deltas = rewards + gamma * next_values * dones - values
        gaes = np.copy(deltas) 
        for t in reversed(range(len(deltas) - 1)):
            update = dones[t] * gamma * lamda * gaes[t + 1]
            gaes[t] = gaes[t] + update

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return  gaes, target

    # TODO use this for training
    # @tf.function
    def predict_chunks(self, model, data, batch_size=64, training=True):
        predictions = []
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            predictions.append(model(batch, training=training))

        return tf.concat(predictions, axis=0)
    
    def reduce_learning_rate(self, ratio: int = 0.95, verbose: bool = True, min_lr: float = 1e-7):
        if self.actor_optimizer.lr.numpy() > min_lr:
            self.actor_optimizer.lr = self.actor_optimizer.lr * ratio
        if self.critic_optimizer.lr.numpy() > min_lr:
            self.critic_optimizer.lr = self.critic_optimizer.lr * ratio

        if verbose: 
            print(f"Reduced learning rate to Actor: {self.actor_optimizer.lr.numpy()} Critic: {self.critic_optimizer.lr.numpy()}")
            
    def custom_logger(self, history: dict) -> dict:

        for key, value in history.items():
            history[key] = np.mean(value)

        history["actor_lr"] = self.actor_optimizer.lr.numpy()
        history["critic_lr"] = self.critic_optimizer.lr.numpy()

        if self.writer:
            with self.writer.as_default():
                for key, value in history.items():
                    tf.summary.scalar(key, value, step = self.epoch)

        history["epoch"] = self.epoch
        self.epoch += 1

        return history

    def train_step_wrapper(func):
        # run function in batches
        # @tf.function
        def wrapper(self, data):
            history = {}
            data = [tf.convert_to_tensor(d) for d in data]
            for _ in range(self.train_epochs):

                for i in range(0, data[0].shape[0], self.batch_size):
                    batch_data = [d[i:i+self.batch_size] for d in data]
                    history_step = func(self, batch_data)

                    for key, value in history_step.items():
                        history[key] = history.get(key, []) + [value.numpy()]

            return self.custom_logger(history)

        return wrapper

    @train_step_wrapper
    # @tf.function
    def train_step(self, data) -> dict:
        states, advantages, old_probs, actions, target = data

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probs = self.actor(states, training=True)  # Forward pass
            values = self.critic(states, training=True)

            # Compute the actor loss value
            # actor_loss, approx_kl_div = self.actor_loss(probs, advantages, old_probs, actions)
            actor_loss = self.actor_loss(probs, advantages, old_probs, actions)
            # Compute the critic loss value
            critic_loss = self.critic_loss(values, target)
            # Compute the entropy loss value
            entropy = self.entropy(probs)

            # Compute actor gradients
            grads_actor = tape1.gradient(actor_loss - entropy, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

            # Compute critic gradients
            grads_critic = tape2.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        return {"a_loss": actor_loss, "c_loss": critic_loss, "entropy": entropy} # , "kl_div": approx_kl_div}

    def train(self, states, actions, rewards, old_probs, dones, next_state) -> dict:
        # reshape memory to appropriate shape for training
        old_probs = np.array(old_probs)
        all_states = np.array(states + [next_state])
        states = np.array(states)
        actions = np.array(actions)

        # Get Critic network predictions 
        all_values = self.predict_chunks(self.critic, all_states, batch_size=self.batch_size, training=False).numpy().squeeze()
        values, next_values = all_values[:-1], all_values[1:]

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, values, next_values)

        
        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar("rewards", np.sum(rewards), step = self.epoch)

        history = self.train_step((states, advantages, old_probs, actions, target))

        return history