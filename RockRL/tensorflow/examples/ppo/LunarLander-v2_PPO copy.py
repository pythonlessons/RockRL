# import random
import gym
import pylab
import numpy as np
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
# from tensorboardX import SummaryWriter
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras import backend as K
import copy
import tensorflow_probability as tfp

def actor_model(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
    X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    output = Dense(action_space, activation="softmax")(X)

    model = Model(inputs = X_input, outputs = output)
    return model

def critic_model(input_shape):
    X_input = Input(input_shape)
    X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
    X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
    value = Dense(1, activation=None)(X)

    model = Model(inputs = X_input, outputs = value)
    return model

class Actor_Model(tf.keras.models.Model):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
    
    def compile(
            self, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
            **kwargs
        ):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def ppo_loss(self, y_pred, advantages, predictions, actions):
        # Defined in https://arxiv.org/abs/1707.06347
        # advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        

        # testing to replace prob
        # argmax = K.argmax(y_pred, axis=1)
        # oh_actions = tf.one_hot(argmax, y_pred.shape[-1])
        # prob = oh_actions * y_pred

        prob = actions * y_pred
        old_prob = actions * predictions

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        states, advantages, predictions, actions = data[0]

        with tf.GradientTape() as tape:
            y_pred = self.actor(states, training=True)  # Forward pass
            # Compute the loss value
            loss = self.ppo_loss(y_pred, advantages, predictions, actions)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}


class Critic_Model(tf.keras.models.Model):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic
    
    def compile(
            self, 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025),
            **kwargs
        ):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def critic_PPO2_loss(self, y_pred, target, values):
        # def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2
        # target, values = y_true[:, :1], y_true[:, 1:2]
        clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
        v_loss1 = (target - clipped_value_loss) ** 2
        v_loss2 = (target - y_pred) ** 2
        
        value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
        #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss
        # return loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        states, target, values = data[0]

        with tf.GradientTape() as tape:
            y_pred = self.critic(states, training=True)  # Forward pass
            # Compute the loss value
            loss = self.critic_PPO2_loss(y_pred, target, values)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}

class _PPOAgent(tf.keras.models.Model):
    def __init__(
        self, 
        ):
        super().__init__()

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 5 # training epochs
        self.shuffle=False
        self.Training_batch = 1000
        #self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0
        # self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.actor = actor_model(input_shape=self.state_size, action_space = self.action_size)
        self.Actor = Actor_Model(actor = self.actor)
        self.Actor.compile(
            optimizer=self.optimizer(learning_rate=self.lr),
            run_eagerly=False
        )

        self.critic = critic_model(input_shape=self.state_size)
        self.Critic = Critic_Model(critic=self.critic)
        self.Critic.compile(
            optimizer=self.optimizer(learning_rate=self.lr),
            run_eagerly=False
        )
        
        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"

        
    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        # prediction = self.Actor.predict(state, verbose=False)[0]
        prediction = self.actor(state, training=False).numpy()[0]
        # dist = tfp.distributions.Categorical(prediction)
        # action = dist.sample().numpy()[0]
        # prediction = self.Actor.predict(state, verbose=False)
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        # prediction = prediction.numpy()[0]
        return action, action_onehot, prediction

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 

        values = self.critic(states, training=False).numpy()
        next_values = self.critic(next_states, training=False).numpy()

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        a_loss = self.Actor.fit(x = (states, advantages, predictions, actions), epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.fit(x = (states, target, values), epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)
        
    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode):
        averaged_episodes = 50
        self.scores_.append(score)
        self.scores = self.scores_[-averaged_episodes:]
        self.episodes_.append(episode)
        self.episodes_ = self.episodes_[-averaged_episodes:]
        self.average_.append(sum(self.scores_[-averaged_episodes:]) / len(self.scores_[-averaged_episodes:]))
        # if str(episode)[-2:] == "00":# much faster than episode % 100
        #     pylab.plot(self.episodes_, self.scores_, 'b')
        #     pylab.plot(self.episodes_, self.average_, 'r')
        #     pylab.title(self.env_name+" PPO training cycle", fontsize=18)
        #     pylab.ylabel('Score', fontsize=18)
        #     pylab.xlabel('Steps', fontsize=18)
        #     try:
        #         pylab.grid(True)
        #         pylab.savefig(self.env_name+".png")
        #     except OSError:
        #         pass
        # saving best models
        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            # self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            # self.lr *= 0.95
            # K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            # K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
            # print(self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    def run(self): # train only when episode is finished
        state, info = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            while not done:
                self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _, info = self.env.step(action)
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done or score<-350:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    # self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    # self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    
                    self.replay(states, actions, rewards, predictions, dones, next_states)

                    state, info = self.env.reset()
                    done, score, SAVING = False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])
                    break

            if self.episode >= self.EPISODES:
                break
        self.env.close()

    # def test(self, test_episodes = 100):
    #     self.load()
    #     for e in range(100):
    #         state = self.env.reset()
    #         state = np.reshape(state, [1, self.state_size[0]])
    #         done = False
    #         score = 0
    #         while not done:
    #             self.env.render()
    #             action = np.argmax(self.Actor.predict(state)[0])
    #             state, reward, done, _ = self.env.step(action)
    #             state = np.reshape(state, [1, self.state_size[0]])
    #             score += reward
    #             if done:
    #                 print("episode: {}/{}, score: {}".format(e, test_episodes, score))
    #                 break
    #     self.env.close()

if __name__ == "__main__":
    env_name = 'LunarLander-v2'
    agent = PPOAgent(env_name)
    agent.run() # train as PPO, train every epesode
    # agent.run_multiprocesses(num_worker = 1)  # train PPO multiprocessed (fastest)