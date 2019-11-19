from imblearn.under_sampling.base import BaseCleaningSampler
import numpy as np


class make_noise(BaseCleaningSampler):
    def __init__(self, noise_level, random_state):
        super().__init__(sampling_strategy='all')
        self.noise_level  = noise_level
        self.random_state = random_state

    def _fit_resample(self, X, y):
        self.mask = np.array(
                [1 for i in range(int(len(y)*self.noise_level))] + \
                [0 for i in range(len(y)-int(len(y)*self.noise_level))]
            ).astype(bool)
        np.random.RandomState(self.random_state)
        np.random.shuffle(self.mask)
        y[self.mask] = np.vectorize(lambda x: 0 if x==1 else 1)(y[self.mask])
        return X, y
