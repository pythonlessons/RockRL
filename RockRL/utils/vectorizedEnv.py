import gym
import numpy as np
import multiprocessing as mp

def run_env(conn, env_name, *kwargs):
    env = gym.make(env_name, *kwargs)

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
            # Assume it's an action
            action = message
            state, reward, done = env.step(action)[:3]
            conn[1].send((state, reward, done))

class VectorizedEnv:
    def __init__(self, env_name: str, num_envs: int=2, *kwargs):
        self.env_name = env_name

        env = gym.make(env_name)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = env._max_episode_steps
        env.close()

        self.conns = [mp.Pipe() for _ in range(num_envs)]
        self.envs = [mp.Process(target=run_env, args=(conn, env_name, *kwargs)) for conn in self.conns]
        for env in self.envs:
            env.start()

    def reset(self, index=None): # return states
        if index is None:
            for conn in self.conns:
                conn[0].send('reset')
            states = np.array([conn[0].recv() for conn in self.conns])
        else:
            self.conns[index][0].send('reset')
            states = self.conns[index][0].recv()

        return states
        
    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn[0].send(action)

        results = [conn[0].recv() for conn in self.conns]
        next_states, rewards, dones = zip(*results)
        
        return np.array(next_states), np.array(rewards), np.array(dones)
    
    def close(self):
        for conn, env in zip(self.conns, self.envs):
            conn[0].send('close')
            env.join()


# class VectorizedEnv:
#     def __init__(self, env_name, num_envs=2, **kwargs):
#         self.envs = [gym.make(env_name, **kwargs) for _ in range(num_envs)]
#         self.num_envs = num_envs
#         self.input_shape = self.envs[0].observation_space.shape
#         self.action_space = self.envs[0].action_space.n
#         self.executor = ThreadPoolExecutor(max_workers=self.num_envs, thread_name_prefix="EnvWorker")

#     def reset(self, index=None):
#         if index is None:
#             state = np.array([env.reset()[0] for env in self.envs])
#             return state
#         else:
#             return self.envs[index].reset()[0]
        
#     def step(self, actions):
#         futures = [self.executor.submit(self.envs[i].step, actions[i]) for i in range(self.num_envs)]
#         results = [future.result()[:3] for future in futures]
#         next_states, rewards, dones = zip(*results)
        
#         return np.array(next_states), np.array(rewards), np.array(dones)