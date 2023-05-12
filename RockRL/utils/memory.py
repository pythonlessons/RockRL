class Memory:
    def __init__(self, input_shape, num_envs=1):
        self.input_shape = input_shape
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
        if states.ndim == len(self.input_shape):
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