import os
import time
import typing
import numpy as np
import tensorflow as tf
from keras import backend as K

class PPOAgent:
    """ Reinforcement Learning agent that learns using Proximal Policy Optimization. """
    def __init__(
        self, 
        actor: tf.keras.models.Model,
        critic: tf.keras.models.Model,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        action_space: str="discrete",
        gamma: float=0.99,
        lamda: float=0.95,
        loss_clipping: float=0.2,
        c1: float=0.5,
        c2: float=0.001,
        train_epochs: int=10,
        batch_size: int=64,
        epoch: int = 0,
        min_max_action: typing.Tuple[float, float] = (-1, 1),
        logdir: str = f"runs/{int(time.time())}",
        writer = None,
        compile: bool=True,
        shuffle: bool=True,
        ):
        """ Reinforcement Learning agent that learns using Proximal Policy Optimization.

        Args:
            actor (tf.keras.models.Model): Actor model
            critic (tf.keras.models.Model): Critic model
            actor_optimizer (tf.keras.optimizers.Optimizer): Optimizer for actor
            critic_optimizer (tf.keras.optimizers.Optimizer): Optimizer for critic
            action_space (str): Action space type. Either 'discrete' or 'continuous'. Defaults to 'discrete'.
            gamma (float): Discount factor for future rewards. Defaults to 0.99.
            lamda (float): Lambda for Generalized Advantage Estimation. Defaults to 0.95.
            loss_clipping (float): Epsilon in clipped loss. Defaults to 0.2.
            c1 (float): Value coefficient. Defaults to 0.5.
            c2 (float): Entropy coefficient. Defaults to 0.001.
            train_epochs (int): Number of epochs to train on each batch. Defaults to 10.
            batch_size (int): Batch size. Defaults to 64.
            epoch (int): Current epoch. Defaults to 0.
            min_max_action (typing.Tuple[float, float]): Min and max values for continuous action space. Defaults to (-1, 1).
            logdir (str): Path to logdir. Defaults to f"runs/{int(time.time())}".
            writer (tf.summary.SummaryWriter): Tensorboard writer. Defaults to None.
            compile (bool): Whether to compile the models. Defaults to True.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
        """
        self.actor = actor
        self.critic = critic
        self.action_space = action_space
        self.gamma = gamma
        self.lamda = lamda
        self.loss_clipping = loss_clipping
        self.c1 = c1
        self.c2 = c2
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.epoch = epoch
        self.min_max_action = min_max_action
        self.shuffle = shuffle

        # Compile the models
        if compile:
            self.actor.compile(optimizer=actor_optimizer)
            self.critic.compile(optimizer=critic_optimizer)

        assert self.action_space in ["discrete", "continuous"], "action_space must be either 'discrete' or 'continuous'"

        # Tensorboard logging
        self.logdir = logdir
        self.writer = writer

        if self.logdir and not writer:
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = tf.summary.create_file_writer(self.logdir)

    def save_models(self, name: str, path: str=None, include_optimizer=True):
        """Save actor and critic models. By default, saves to self.logdir.
        
        Args:
            name (str): Name of the model
            path (str, optional): Path to save the model. Defaults to None.
            include_optimizer (bool, optional): Whether to save the optimizer state. Defaults to True.
        """
        path = path or self.logdir
        self.actor.save(path + f"/{name}_actor.h5", include_optimizer=include_optimizer)
        self.critic.save(path + f"/{name}_critic.h5", include_optimizer=include_optimizer)

    def critic_loss(self, y_pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """ Mean Squared Error loss for critic"""
        y_pred = tf.squeeze(y_pred)
        target = tf.squeeze(target)
        loss = K.mean((y_pred - target) ** 2)
        return loss
    
    def logprob_dist(self, probs: tf.Tensor, actions: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """ Compute log probability of actions given the distribution, and the entropy of the distribution.
        This is used for discrete action spaces.
        """
        oh_actions = K.one_hot(tf.cast(actions, tf.uint8), probs.shape[-1])
        probs = K.sum(oh_actions * probs, axis=1)
        log_probs = K.log(probs + 1e-10)

        entropy = -K.mean(probs * K.log(probs + 1e-10))

        return log_probs, entropy
    
    def logprob_dist_continuous(self, probs: tf.Tensor, actions: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """ Compute log probability of actions given the distribution, and the entropy of the distribution.
        This is used for continuous action spaces.
        """
        actions = tf.cast(actions, tf.float32)
        # Get probs size from the last dimension of probs, because in actor model, we concatenate mu and sigma
        probs_size = tf.cast(probs.shape[-1] / 2, tf.int32)
        mu, sigma = probs[:, :probs_size], probs[:, probs_size:]

        # compute the exponent of the gaussian dist
        exponent = -0.5 * K.square(actions - mu) / K.square(sigma)
        # compute log coefficient
        log_coeff = -0.5 * K.log(2.0 * np.pi * K.square(sigma))
        # compute log probability
        log_prob = log_coeff + exponent
        log_prob = tf.reduce_sum(log_prob, axis=1)

        # compute entropy
        entropy_coeff = 0.5 * K.log(2.0 * np.pi * np.e * K.square(sigma))
        entropy = tf.reduce_sum(entropy_coeff, axis=1)
        entropy = -K.mean(entropy)

        return log_prob, entropy

    def actor_loss(self, probs, advantages, old_probs, actions) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """ Compute actor loss. Returns loss, entropy, and approx_kl_divergence.
        PPO actor loss defined in https://arxiv.org/abs/1707.06347
        """
        if self.action_space == "discrete":
            log_prob, entropy = self.logprob_dist(probs, actions)
            log_old_prob, _ = self.logprob_dist(old_probs, actions)

        elif self.action_space == "continuous":
            log_prob, entropy = self.logprob_dist_continuous(probs, actions)
            log_old_prob, _ = self.logprob_dist_continuous(old_probs, actions)

        log_ratio = log_prob - log_old_prob
        ratio = K.exp(log_ratio)
        
        p1 = advantages * ratio
        p2 = advantages * K.clip(ratio, 1-self.loss_clipping, 1+self.loss_clipping)

        loss = -K.mean(K.minimum(p1, p2))

        approx_kl_divergence = K.mean((K.exp(log_ratio) - 1) - log_ratio)

        return loss, entropy, approx_kl_divergence

    def act(self, state: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        state_dim = state.ndim
        if state_dim < len(self.actor.input_shape):
            state = np.expand_dims(state, axis=0)

        # Use the network to predict the next action to take, using the model
        probs = self.actor(state, training=False).numpy()
        if self.action_space == "discrete":
            actions = np.array([np.random.choice(prob.shape[0], p=prob) for prob in probs])

        elif self.action_space == "continuous":
            # in continuous action space, the network outputs mean and sigma should be concatenated
            probs_size = int(probs.shape[-1] / 2)
            a_probs, sigma = probs[:, :probs_size], probs[:, probs_size:]
            actions = np.random.normal(a_probs, sigma)
    
        if state_dim < len(self.actor.input_shape):
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

    def get_gaes(self, rewards, dones, values, next_values, gamma: float = 0.99, lamda: float = 0.9, normalize: bool=True):
        values = np.array(values, dtype=np.float32)
        next_values = np.array(next_values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = 1 - np.array(dones, dtype=np.float32)
        gaes = rewards + gamma * next_values * dones - values
        # gaes = np.copy(deltas) 
        for t in reversed(range(len(gaes) - 1)):
            update = dones[t] * gamma * lamda * gaes[t + 1]
            gaes[t] = gaes[t] + update

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return  gaes, target

    def predict_chunks(
            self, 
            model: tf.keras.models.Model, 
            data: typing.Union[np.ndarray, tf.Tensor],
            batch_size: int=64, 
            training: bool=True
        ) -> tf.Tensor:
        """ Predict data in chunks to avoid OOM error.

        Args:
            model (tf.keras.models.Model): Model to predict data with.
            data (typing.Union[np.ndarray, tf.Tensor]): Data to predict.
            batch_size (int, optional): Batch size to use. Defaults to 64.
            training (bool, optional): Whether to use training mode. Defaults to True.
        """
        predictions = []
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            predictions.append(model(batch, training=training))

        return tf.concat(predictions, axis=0)
    
    def reduce_learning_rate(self, ratio: int = 0.95, verbose: bool = True, min_lr: float = 1e-06):
        if float(self.actor.optimizer.lr.numpy()) > min_lr:
            self.actor.optimizer.lr = self.actor.optimizer.lr * ratio
            if verbose: 
                print(f"Reduced learning rate to Actor: {self.actor.optimizer.lr.numpy()}")
        if float(self.critic.optimizer.lr.numpy()) > min_lr:
            self.critic.optimizer.lr = self.critic.optimizer.lr * ratio
            if verbose: 
                print(f"Reduced learning rate to Critic: {self.critic.optimizer.lr.numpy()}")

            
    def custom_logger(self, history: dict) -> dict:
        for key, value in history.items():
            history[key] = np.mean(value)

        history["actor_lr"] = self.actor.optimizer.lr.numpy()
        history["critic_lr"] = self.critic.optimizer.lr.numpy()

        if self.writer:
            with self.writer.as_default():
                for key, value in history.items():
                    tf.summary.scalar(key, value, step = self.epoch)

        history["epoch"] = self.epoch
        self.epoch += 1

        return history

    def train_step_wrapper(func):
        """ Decorator to wrap train_step function. Used to train the model for a given number of epochs and batches.
        Then collects the history and logs it to tensorboard.
        """
        def wrapper(self, data):
            history = {}
            for _ in range(self.train_epochs):

                for i in range(0, data[0].shape[0], self.batch_size):
                    batch_data = [d[i:i+self.batch_size] for d in data]
                    history_step = func(self, batch_data)

                    for key, value in history_step.items():
                        history[key] = history.get(key, []) + [value.numpy()]

            return self.custom_logger(history)

        return wrapper

    @train_step_wrapper
    @tf.function
    def train_step(self, data) -> dict:
        states, advantages, old_probs, actions, target = data

        if self.shuffle:
            indices = tf.range(0, states.shape[0])
            indices = tf.random.shuffle(indices)
            states = tf.gather(states, indices)
            advantages = tf.gather(advantages, indices)
            old_probs = tf.gather(old_probs, indices)
            actions = tf.gather(actions, indices)
            target = tf.gather(target, indices)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probs = self.actor(states, training=True)  # Forward pass
            values = self.critic(states, training=True)

            # Compute the actor loss value
            actor_loss, entropy, approx_kl_div = self.actor_loss(probs, advantages, old_probs, actions)
            # Compute the critic loss value
            critic_loss = self.critic_loss(values, target) * self.c1
            # Compute the entropy loss value
            entropy_loss = entropy * self.c2

            # Compute actor gradients
            grads_actor = tape1.gradient(actor_loss + entropy_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

            # Compute critic gradients
            grads_critic = tape2.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        return {"a_loss": actor_loss, "c_loss": critic_loss, "entropy": entropy, "kl_div": approx_kl_div}

    def train(self, states, actions, rewards, old_probs, dones, next_state) -> dict:
        # reshape memory to appropriate shape for training
        old_probs = np.array(old_probs)
        # all_states = np.concatenate([states, [next_state]], axis=0)
        all_states = np.array(states + [next_state])
        states = np.array(states)
        actions = np.array(actions)

        # Get Critic network predictions 
        all_values = self.predict_chunks(self.critic, all_states, batch_size=self.batch_size, training=False).numpy().squeeze()
        values, next_values = all_values[:-1], all_values[1:]

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, values, next_values, gamma=self.gamma, lamda=self.lamda, normalize=True)
        
        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar("rewards", np.sum(rewards), step = self.epoch)

        # shuffle = np.arange(states.shape[0])
        # np.random.shuffle(shuffle)
        # states = states[shuffle]
        # advantages = advantages[shuffle]
        # old_probs = old_probs[shuffle]
        # actions = actions[shuffle]
        # target = target[shuffle]

        history = self.train_step((states, advantages, old_probs, actions, target))

        # Clear TensorFlow sessions
        # tf.keras.backend.clear_session()

        return history