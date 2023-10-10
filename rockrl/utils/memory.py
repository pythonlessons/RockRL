import numpy as np

class Memory:
    """ Memory class for storing the experiences of the agent
    """
    def __init__(self):
        self._reset = True
        
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.predictions = []
        self.terminateds = []
        self.truncateds = []
        self.next_state = None
        self.infos = []

        self._reset = False

    def __len__(self):
        return len(self.rewards)
    
    @property
    def score(self):
        return np.sum(self.rewards)

    def append(self, state, action, reward, prediction, terminated, truncated, next_state, info: dict={}):
        if self._reset:
            self.reset()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.predictions.append(prediction)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.next_state = next_state
        self.infos.append(info)

    def get(self, reset=True):
        self._reset = reset

        return self.states, self.actions, self.rewards, self.predictions, self.terminateds, self.truncateds, self.next_state, self.infos
    
    @property
    def done(self):
        return self.terminateds[-1] or self.truncateds[-1]

class MemoryManager:
    """ Memory class for storing the experiences of the agent in multiple environments
    """
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.memory = [Memory() for _ in range(num_envs)]

    def append(self, *kwargs):
        data = list(zip(*kwargs))
        for i in range(self.num_envs):
            self.memory[i].append(*data[i])

    def __getitem__(self, index: int):
        return self.memory[index]

    def done_indices(self):
        return [env for env in range(self.num_envs) if self.memory[env].done]