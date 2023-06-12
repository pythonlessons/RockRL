import numpy as np
import multiprocessing as mp

class CustomEnv:
    def __init__(self, custom_env_object, os_hist_steps=4, **kwargs) -> None:
        self.env = custom_env_object(**kwargs)
        self.os_hist_steps = os_hist_steps
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps

        self.observation_space = np.zeros((self.os_hist_steps, self.env.observation_space.shape[0]))

    def reset(self):
        self.state = np.zeros((self.os_hist_steps, self.env.observation_space.shape[0]))
        state = self.env.reset()[0]
        self.state[-1] = state
        return np.array(self.state), {}
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)[:4]
        self.state = np.roll(self.state, shift=-1, axis=0)
        self.state[-1] = next_state
        return np.array(self.state), reward, done, info
    
    def close(self):
        self.env.close()

    def render(self, **kwargs):
        return self.env.render(**kwargs)


def run_env(conn, env_object, kwargs):
    env = env_object(**kwargs)

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
            elif message == 'render':
                results = env.render()
                results = results if results is not None else []
                conn[1].send(results)

        else:
            # Assume it's an action
            action = message
            state, reward, done = env.step(action)[:3]
            conn[1].send((state, reward, done))


class VectorizedEnv:
    def __init__(self, env_object, num_envs: int=2, **kwargs):

        env = env_object(**kwargs)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = env._max_episode_steps
        env.close()

        self.conns = [mp.Pipe() for _ in range(num_envs)]
        self.envs = [mp.Process(target=run_env, args=(conn, env_object, kwargs)) for conn in self.conns]
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
    
    def render(self, index=None):
        if index is None:
            for conn in self.conns:
                conn[0].send('render')
            results = [conn[0].recv() for conn in self.conns]
        else:
            self.conns[index][0].send('render')
            results = self.conns[index][0].recv()

        return results

    def close(self):
        for conn, env in zip(self.conns, self.envs):
            conn[0].send('close')
            env.join()
            env.close()