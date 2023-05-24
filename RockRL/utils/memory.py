import numpy as np
import gc

class Memory:
    """ Memory class for storing the experiences of the agent in multiple environments
    """
    def __init__(self, input_shape, num_envs=1):
        self.input_shape = input_shape
        self.num_envs = num_envs
        self.reset()

    # def reset(self, index=None):
    #     if index is None:
    #         self.states = [np.empty((0,) + self.input_shape) for _ in range(self.num_envs)]
    #         self.actions = [np.empty((0,)) for _ in range(self.num_envs)]
    #         self.rewards = [np.empty((0,)) for _ in range(self.num_envs)]
    #         self.predictions = [np.empty((0,)) for _ in range(self.num_envs)]
    #         self.dones = [np.empty((0,), dtype=bool) for _ in range(self.num_envs)]
    #         self.next_states = [None for _ in range(self.num_envs)]
    #     else:
    #         self.states[index] = np.empty((0,) + self.input_shape)
    #         self.actions[index] = np.empty((0,))
    #         self.rewards[index] = np.empty((0,))
    #         self.predictions[index] = np.empty((0,))
    #         self.dones[index] = np.empty((0,), dtype=bool)
    #         self.next_states[index] = None

    # def append(self, states, actions, rewards, predictions, dones, next_states):
    #     # check whether the input is dimensioned for one or multiple environments
    #     if states.ndim == len(self.input_shape):
    #         states, actions, rewards, predictions, dones, next_states = [np.expand_dims(x, axis=0) for x in [states, actions, rewards, predictions, dones, next_states]]

    #     for i in range(self.num_envs):
    #         self.states[i] = np.concatenate([self.states[i], [states[i]]], axis=0)
    #         self.actions[i] = np.concatenate([self.actions[i], [actions[i]]], axis=0)
    #         self.rewards[i] = np.concatenate([self.rewards[i], rewards[i]], axis=0)
    #         self.predictions[i] = np.concatenate([self.predictions[i], predictions[i]], axis=0)
    #         self.dones[i] = np.concatenate([self.dones[i], dones[i]], axis=0)
    #         self.next_states[i] = next_states[i]

    # def reset(self, index=None):
    #     if index is None:
    #         self.states = [np.empty((0,) + self.input_shape) for _ in range(self.num_envs)]
    #         self.actions = [[] for _ in range(self.num_envs)]
    #         self.rewards = [[] for _ in range(self.num_envs)]
    #         self.predictions = [[] for _ in range(self.num_envs)]
    #         self.dones = [[] for _ in range(self.num_envs)]
    #         self.next_states = [None for _ in range(self.num_envs)]

    #     else:
    #         self.states[index] = np.empty((0,) + self.input_shape)
    #         self.actions[index] = []
    #         self.rewards[index] = []
    #         self.predictions[index] = []
    #         self.dones[index] = []
    #         self.next_states[index] = None

    # def append(self, states, actions, rewards, predictions, dones, next_states):
    #     # check whether the input is dimensioned for one or multiple environments
    #     if states.ndim == len(self.input_shape):
    #         states, actions, rewards, predictions, dones, next_states = [[x] for x in [states, actions, rewards, predictions, dones, next_states]]

    #     for i in range(self.num_envs):
    #         self.states[i] = np.concatenate([self.states[i], [states[i]]], axis=0)
    #         self.actions[i].append(actions[i])
    #         self.rewards[i].append(rewards[i])
    #         self.predictions[i].append(predictions[i])
    #         self.dones[i].append(dones[i])
    #         self.next_states[i] = next_states[i]

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

        gc.collect()

    def append(self, states, actions, rewards, predictions, dones, next_states):
        # check whether the input is dimensioned for one or multiple environments
        if states.ndim == len(self.input_shape):
            states, actions, rewards, predictions, dones, next_states = [[x] for x in [states, actions, rewards, predictions, dones, next_states]]

        for i in range(self.num_envs):
            self.states[i].append(states[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.predictions[i].append(predictions[i])
            self.dones[i].append(dones[i])
            self.next_states[i] = next_states[i]

    def get(self, index: int=None):
        if self.num_envs == 1:
            return self.states[0], self.actions[0], self.rewards[0], self.predictions[0], self.dones[0], self.next_states[0]
        
        if index is None:
            raise ValueError("Must provide indexes when using multiple environments")
        return self.states[index], self.actions[index], self.rewards[index], self.predictions[index], self.dones[index], self.next_states[index]
    
    def lengths(self) -> np.ndarray:
        """ Returns the lengths of the episodes in the memory for each environment"""
        return np.array([len(r) for r in self.rewards]).astype(np.int32)
    
    def done_indices(self, max_episode_steps: int = None) -> list:
        """ Returns the indices of the environments that are done (either by reaching the max episode steps or by the environment itself)"""
        done_indices = np.array([env for env in range(self.num_envs) if self.dones[env][-1]]).astype(np.int32)
        max_episode_envs = np.where(self.lengths() >= max_episode_steps)[0]
        if max_episode_envs.any():
            done_indices = np.unique(np.concatenate((done_indices, max_episode_envs))).astype(np.int32)

        return done_indices.tolist()