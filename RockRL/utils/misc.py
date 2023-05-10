import numpy as np

class MeanAverage:
    def __init__(
        self, 
        window_size: int=50,
        best_mean_score: float=-np.inf,
        best_mean_score_episode: int=100,
        best_wait: int = 50,
        ):
        self.window_size = window_size
        self.best_mean_score = best_mean_score
        self.best_mean_score_episode = best_mean_score_episode
        self.best_wait = best_wait

        self.values = []
        self.mean_score = None

    def __call__(self, x):
        self.values.append(x)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        self.mean_score = np.mean(self.values)

        return self.mean_score
    
    def is_best(self, episode):
        if self.mean_score > self.best_mean_score and episode > self.best_mean_score_episode:
            self.best_mean_score = self.mean_score
            self.best_mean_score_episode = episode
            return True
        
        return False

    def is_improoving(self, episode):
        if self.best_mean_score_episode + self.best_wait < episode:
            self.best_mean_score_episode = episode
            return False
        
        return True