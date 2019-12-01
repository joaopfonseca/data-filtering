from imblearn.under_sampling.base import BaseCleaningSampler
import numpy as np
from copy import deepcopy

class make_binary_noise(BaseCleaningSampler):
    def __init__(self, noise_level=.1, random_state=None):
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


class make_multiclass_noise(BaseCleaningSampler):
    def __init__(self, transfer_map=None, noise_level=.1, random_state=None):
        super().__init__(sampling_strategy='all')
        self.noise_level  = noise_level
        self.random_state = random_state
        self.transfer_map = transfer_map

    def _fit_resample(self, X, y):
        _X, _y = deepcopy(X), deepcopy(y)
        # set transfer_map
        if self.transfer_map is None:
            np.random.seed(self.random_state)
            self.transfer_map = {
                k:np.random.randint(0,np.unique(_y).shape)[0]
                for k in np.unique(_y)
            }

        self.mask = np.zeros(_y.shape)
        for label in self.transfer_map.keys():
            size = len(_y[_y==label])
            _mask = np.array(
                    [1 for i in range(int(size*self.noise_level))] + \
                    [0 for i in range(size-int(size*self.noise_level))]
                ).astype(bool)

            np.random.seed(self.random_state)
            np.random.shuffle(_mask)
            self.mask[_y==label] = _mask

        self.mask = self.mask.astype(bool)
        _y[self.mask] = np.vectorize(lambda x: self.transfer_map[x])(_y[self.mask])
        return _X, _y
