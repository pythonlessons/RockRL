import os
import yaml
import time
import typing
import numpy as np
import tensorflow as tf
from keras import backend as K

from rockrl.utils.memory import Memory

import threading
import queue

class TensorBoardLogger:
    def __init__(self, writer, epoch=0):
        self.writer = writer
        self.epoch = epoch
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._log_worker)
        self.worker_thread.start()

    def log_to_queue(self, data, epoch):
        self.epoch = epoch
        self.log_queue.put(data)

    def _log_worker(self):
        while not self.stop_event.is_set():
            try:
                data = self.log_queue.get(timeout=1)  # Use timeout to make it non-blocking
            except queue.Empty:
                continue
            
            if self.writer:
                with self.writer.as_default():
                    for key, value in data.items():
                        tf.summary.scalar(key, value, step=self.epoch)
                    self.writer.flush()

    def stop_worker(self):
        self.stop_event.set()
        self.worker_thread.join()


class PPOAgent:
    """ Reinforcement Learning agent that learns using Proximal Policy Optimization. """
    def __init__(
        self, 
        actor: tf.keras.models.Model,
        critic: tf.keras.models.Model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        action_space: str="discrete",
        gamma: float=0.99,
        lamda: float=0.95,
        loss_clipping: float=0.2,
        c1: float=0.5,
        c2: float=0.01,
        train_epochs: int=10,
        train_epochs_annealing: bool=True,
        batch_size: int=64,
        epoch: int = 0,
        logdir: str = f"runs/{int(time.time())}",
        writer = None,
        writer_comment: str = "",
        shuffle: bool=True,
        kl_coeff: float=0.2,
        grad_clip_value: float=0.5,
        ):
        """ Reinforcement Learning agent that learns using Proximal Policy Optimization.

        Args:
            actor (tf.keras.models.Model): Actor model
            critic (tf.keras.models.Model): Critic model
            optimizer (tf.keras.optimizers.Optimizer): Optimizer for actor and critic models. Defaults to tf.keras.optimizers.Adam(learning_rate=0.0001).
            action_space (str): Action space type. Either 'discrete' or 'continuous'. Defaults to 'discrete'.
            gamma (float): Discount factor for future rewards. Defaults to 0.99.
            lamda (float): Lambda for Generalized Advantage Estimation. Defaults to 0.95.
            loss_clipping (float): Epsilon in clipped loss. Defaults to 0.2.
            c1 (float): Value coefficient. Defaults to 0.5.
            c2 (float): Entropy coefficient. Defaults to 0.01.
            train_epochs (int): Number of epochs to train on each batch. Defaults to 10.
            train_epochs_annealing (bool): Whether to anneal the learning rate for the number of epochs to train on each batch. Defaults to True.
            batch_size (int): Batch size. Defaults to 64.
            epoch (int): Current epoch. Defaults to 0.
            logdir (str): Path to logdir. Defaults to f"runs/{int(time.time())}".
            writer (tf.summary.SummaryWriter): Tensorboard writer. Defaults to None.
            writer_comment (str): Comment for Tensorboard writer. Defaults to "".
            compile (bool): Whether to compile the models. Defaults to True.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
            kl_coeff (float): KL divergence coefficient, to prevent the policy from changing too much. Defaults to 0.2.
            grad_clip_value (float): Value to clip gradients to prevent exploding gradients. Defaults to 0.5.
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
        self.train_epochs_annealing = train_epochs_annealing
        self.batch_size = batch_size
        self.epoch = epoch
        self.shuffle = shuffle
        self.kl_coeff = kl_coeff
        self.grad_clip_value = grad_clip_value

        self.optimizer = optimizer
        self.learning_rate = float(self.optimizer.lr.numpy())

        assert self.action_space in ["discrete", "continuous"], "action_space must be either 'discrete' or 'continuous'"

        # Tensorboard logging
        self.logdir = logdir
        self.writer = writer
        self.writer_comment = writer_comment

        if self.logdir and not writer:
            os.makedirs(self.logdir, exist_ok=True)
            self.writer = tf.summary.create_file_writer(self.logdir, filename_suffix=self.writer_comment)

        self.tensorBoardLogger = TensorBoardLogger(self.writer, self.epoch)
        tf.get_logger().setLevel('ERROR')
        self.save_config()

    def save_config(self):
        """Save the agent's config to self.logdir/config.yml"""
        config = {}
        for key, value in self.__dict__.items():
            if key in ["actor", "critic", "optimizer", "writer"]:
                continue

            config[key] = value

        if self.logdir:
            config_file = self.logdir + "/config.yaml"
            if not os.path.exists(config_file):
                try:
                    with open(config_file, "w") as f:
                        yaml.dump(config, f)
                except:
                    print("Failed to save config to file")

        # save config to tensorboard, for easy viewing
        if self.writer:
            with self.writer.as_default():
                for key, value in config.items():
                    tf.summary.text(key, str(value), step=0)

    def save_models(self, name: str, path: str=None, include_optimizer=False):
        """Save actor and critic models. By default, saves to self.logdir.
        
        Args:
            name (str): Name of the model
            path (str, optional): Path to save the model. Defaults to None.
            include_optimizer (bool, optional): Whether to save the optimizer state. Defaults to True.
        """
        path = path or self.logdir
        if path:
            self.actor.save(path + f"/{name}_actor.h5", include_optimizer=include_optimizer)
            self.critic.save(path + f"/{name}_critic.h5", include_optimizer=include_optimizer)

    def critic_loss(self, y_pred: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """ Mean Squared Error loss for critic"""
        y_pred = tf.squeeze(y_pred)
        target = tf.squeeze(target)

        v_loss_unclipped = (y_pred - target) ** 2
        v_clipped = target + K.clip(
            y_pred - target,
            -self.loss_clipping,
            self.loss_clipping
        )
        v_loss_clipped = (v_clipped - target) ** 2
        v_loss_max = K.maximum(v_loss_unclipped, v_loss_clipped)
        loss = K.mean(v_loss_max)

        return loss
    
    def logprob_dist(self, probs: tf.Tensor, actions: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """ Compute log probability of actions given the distribution, and the entropy of the distribution.
        This is used for discrete action spaces.
        """
        oh_actions = K.one_hot(tf.cast(actions, tf.uint8), probs.shape[-1])
        probs = K.sum(oh_actions * probs, axis=1)
        log_probs = K.log(probs + 1e-10)

        entropy = -K.mean(probs * K.log(probs + 1e-10))
        sigma = 1 - K.mean(probs)

        return log_probs, entropy, sigma
    
    def logprob_dist_continuous(self, probs: tf.Tensor, actions: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """ Compute log probability of actions given the distribution, and the entropy of the distribution.
        This is used for continuous action spaces.
        """
        actions = tf.cast(actions, tf.float32)
        # Get probs size from the last dimension of probs, because in actor model, we concatenate mu and sigma
        mu, sigma = probs[:, :-1], probs[:, -1]
        sigma = tf.expand_dims(sigma, -1)

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
        entropy = K.mean(entropy)

        return log_prob, entropy, sigma

    def actor_loss(self, probs, advantages, old_probs, actions) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """ Compute actor loss. Returns loss, entropy, and approx_kl_divergence.
        PPO actor loss defined in https://arxiv.org/abs/1707.06347
        """
        if self.action_space == "discrete":
            log_prob, entropy, sigma = self.logprob_dist(probs, actions)
            log_old_prob, _, _ = self.logprob_dist(old_probs, actions)

        elif self.action_space == "continuous":
            log_prob, entropy, sigma = self.logprob_dist_continuous(probs, actions)
            log_old_prob, _, _ = self.logprob_dist_continuous(old_probs, actions)
            sigma = K.mean(sigma)

        log_ratio = log_prob - log_old_prob
        ratio = K.exp(log_ratio)
        
        p1 = advantages * ratio
        p2 = advantages * K.clip(ratio, 1-self.loss_clipping, 1+self.loss_clipping)

        loss = -K.mean(K.minimum(p1, p2))

        approx_kl_divergence = K.mean((ratio - 1) - log_ratio)

        return loss, entropy, approx_kl_divergence, sigma

    def act(self, state: np.ndarray, training: bool = True) -> typing.Tuple[np.ndarray, np.ndarray]:
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        state_dim = state.ndim
        if state_dim < len(self.actor.input_shape):
            state = np.expand_dims(state, axis=0)

        # Use the network to predict the next action to take, using the model
        probs = self.actor(state, training=training).numpy()
        if self.action_space == "discrete":
            # in discrete action space, the network outputs a probability distribution over the actions
            if training:
                actions = np.array([np.random.choice(prob.shape[0], p=prob) for prob in probs])
            else:
                actions = np.argmax(probs, axis=1)

        elif self.action_space == "continuous":
            # in continuous action space, the network outputs mean and sigma should be concatenated
            a_probs, sigma = probs[:, :-1], probs[:, -1]
            if training:
                actions = np.random.normal(a_probs, np.expand_dims(sigma, -1))
            else:
                actions = a_probs
    
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
        if self.learning_rate > min_lr:
            self.learning_rate *= ratio
            self.optimizer.lr = self.learning_rate
            if verbose: 
                print(f"Reduced learning rate to {self.learning_rate}")

    def log_to_writer(self, data: dict, epoch: int = None):
        epoch = epoch or self.epoch
        self.tensorBoardLogger.log_to_queue(data, epoch)

    def custom_logger(self, history: dict) -> dict:
        for key, value in history.items():
            history[key] = np.mean(value)

        history["lr"] = self.optimizer.lr.numpy()

        self.log_to_writer(history)

        history["epoch"] = self.epoch
        self.epoch += 1

        return history

    def train_step_wrapper(func):
        """ Decorator to wrap train_step function. Used to train the model for a given number of epochs and batches.
        Then collects the history and logs it to tensorboard.
        """
        def wrapper(self, data):
            history = {}
            for epoch in range(self.train_epochs):

                if self.train_epochs_annealing:
                    # Reduce learning rate for each epoch in total number of epochs
                    frac = 1.0 - epoch  / self.train_epochs
                    self.optimizer.lr = self.learning_rate * frac

                for i in range(0, data[0].shape[0], self.batch_size):
                    batch_data = [d[i:i+self.batch_size] for d in data]
                    history_step = func(self, batch_data) # call train_step function on batch data

                    for key, value in history_step.items():
                        history[key] = history.get(key, []) + [value.numpy()]

            return self.custom_logger(history)

        return wrapper

    def clip_gradients(self, gradients: list, grad_clip_value: float = 0.5) -> list:
        if grad_clip_value is None or not isinstance(grad_clip_value, float):
            return gradients
        # Defuse inf gradients (due to super large losses). This is a hack to make the model more stable.
        cliped_grads, _ = tf.clip_by_global_norm(gradients, grad_clip_value)
        # If the global_norm is inf -> All grads will be NaN. Stabilize this
        # here by setting them to 0.0. This will simply ignore destructive loss
        # calculations.
        grads_no_nan = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) for g in cliped_grads]

        return grads_no_nan
    
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

        with tf.GradientTape() as tape:
            probs = self.actor(states, training=True)  # Forward pass
            values = self.critic(states, training=True)

            # Compute the actor loss value
            actor_loss, entropy, approx_kl_div, sigma = self.actor_loss(probs, advantages, old_probs, actions)
            # Compute the critic loss value
            critic_loss = self.critic_loss(values, target) * self.c1
            # Compute the entropy loss value
            entropy_loss = entropy * self.c2

            # Compute the total loss value
            # summing actor_loss, critic_loss and entropy_loss to get loss
            # adding approx_kl_div to loss to prevent old probs getting too far from new probs
            kl_error = approx_kl_div * self.kl_coeff if approx_kl_div >= self.kl_coeff else 0.0
            loss = actor_loss + entropy_loss + critic_loss + kl_error
            grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
            grads = self.clip_gradients(grads, self.grad_clip_value)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        results = {"loss": loss, "a_loss": actor_loss, "c_loss": critic_loss, "entropy": entropy, "kl_div": approx_kl_div, "sigma": sigma}

        return results

    def train(self, memory: Memory) -> dict:
        if not isinstance(memory, Memory):
            raise TypeError("memory must be an instance of Memory object")

        # Get and reshape memory to appropriate shape for training
        states, actions, rewards, old_probs, dones, truncateds, next_state, infos = memory.get()
        old_probs = np.array(old_probs)
        all_states = np.array(states + [next_state])
        states = np.array(states)
        actions = np.array(actions)

        # Get Critic network predictions 
        all_values = self.predict_chunks(self.critic, all_states, batch_size=self.batch_size, training=False).numpy().squeeze()
        values, next_values = all_values[:-1], all_values[1:]

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, values, next_values, gamma=self.gamma, lamda=self.lamda, normalize=True)
        
        self.log_to_writer({"rewards": np.sum(rewards)})

        history = self.train_step((states, advantages, old_probs, actions, target))

        return history
    
    def summary(self, actor=True, critic=True):
        if actor: 
            self.actor.summary()
        if critic:
            self.critic.summary()

    def close(self):
        self.tensorBoardLogger.stop_worker()
        self.writer.close()