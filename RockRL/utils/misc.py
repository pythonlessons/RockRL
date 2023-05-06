import numpy as np

class MeanAverage:
    def __init__(self, window_size: int=50):
        self.window_size = window_size
        self.values = []

    def __call__(self, x):
        self.values.append(x)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        return np.mean(self.values)