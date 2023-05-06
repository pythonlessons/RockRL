import numpy as np
import tensorflow as tf
from keras import backend as K

# import tfp
import tensorflow_probability as tfp

class PPOAgent(tf.keras.models.Model):
    def __init__(
        self, 
        actor,
        critic,
        loss_clipping=0.2,
        c1=0.5,
        c2=0.001,
        ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.loss_clipping = loss_clipping # epsilon in clipped loss
        self.c1 = c1 # value coefficient
        self.c2 = c2 # entropy coefficient

    def compile(
        self, 
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        **kwargs
        ):
        super().compile(**kwargs)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def entropy(self, y_pred):
        return -K.mean(y_pred * K.log(y_pred + 1e-10), axis=1) * self.c2

    def critic_loss(self, y_pred, target):
        # loss = self.c1 * K.mean((target - y_pred) ** 2)
        loss = self.c1 * K.mean((y_pred - target) ** 2)
        return loss
    
    def actor_loss(self, y_pred, advantages, predictions, actions):
        # Defined in https://arxiv.org/abs/1707.06347

        dist = tfp.distributions.Categorical(y_pred)
        prob = dist.log_prob(actions)

        dist_old = tfp.distributions.Categorical(predictions)
        old_prob = dist_old.log_prob(actions)

        ratio = K.exp(prob - old_prob)

        advantages = tf.squeeze(advantages)



        # # testing to replace prob
        # # argmax = K.argmax(y_pred, axis=1)
        # # oh_actions = tf.one_hot(argmax, y_pred.shape[-1])
        # # prob = oh_actions * y_pred

        # actions_onehot = K.one_hot(actions, y_pred.shape[-1])

        # prob = actions_onehot * y_pred
        # old_prob = actions_onehot * predictions

        # prob = K.clip(prob, 1e-10, 1.0)
        # old_prob = K.clip(old_prob, 1e-10, 1.0)

        # ratio = K.exp(K.log(prob) - K.log(old_prob))
        
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
        return np.vstack(gaes), np.vstack(target)

    # TODO use this for training
    def predict_chunks(self, model, data, batch_size=64, training=True):
        predictions = []
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            predictions.append(model(batch, training=training))

        return tf.concat(predictions, axis=0)

    # @tf.function
    def train_step(self, data):
        # states, predictions, actions, rewards, dones, next_state = data # [0]
        states, advantages, predictions, actions, target, rewards, dones, values, next_values = data[0]

        # # numpy to tensor
        # # states = tf.convert_to_tensor(states, dtype=tf.float32)
        # # next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        # next_state = tf.expand_dims(next_state, axis=0)
        # combined_states = tf.concat([states, next_state], axis=0)
        # # next_states = combined_states[1:]

        # values_pred = self.predict_chunks(self.critic, combined_states, batch_size=64, training=False)

        # values = tf.squeeze(values_pred[:-1])
        # next_values = tf.squeeze(values_pred[1:])

        # advantages, target = self.get_gaes_tf(rewards, dones, values, next_values)

        # advantages, target = self.get_gaes_tf(rewards, dones, values, next_values)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            probs = self.actor(states, training=True)  # Forward pass
            # probs = self.predict_chunks(self.actor, states, batch_size=64, training=True)  # Forward pass

            # values_pred = self.critic(states, training=True)
            # values_pred = self.critic(combined_states, training=True)

            # values = self.predict_chunks(self.critic, states, batch_size=64, training=True)
            values = self.critic(states, training=True)

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
        self.epochs = 10
        # self.shuffle = False
        # reshape memory to appropriate shape for training
        predictions = np.vstack(predictions)

        # # Get Critic network predictions 
        # # append next state to states
        all_states = np.array(states + [next_state])
        # # next_states = np.array(all_states[1:])
        all_values = self.critic(all_states, training=False).numpy().squeeze()
        values, next_values = all_values[:-1], all_values[1:]

        # # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, values, next_values)
        # # advantages, target = self.get_gaes_tf(rewards, dones, values, next_values)

        states = np.array(states)
        actions = np.array(actions)

        rewards = np.array(rewards)
        dones = np.array(dones)

        # self.train_step((states, advantages, predictions, actions, target, rewards, dones, values, next_values, next_state))
        # for _ in range(self.epochs):
        #     self.train_step((states, predictions, actions, rewards, dones, next_state))

        self.fit(x=(states, advantages, predictions, actions, target, rewards, dones, values, next_values), epochs=self.epochs, shuffle=False, verbose=False)