from imblearn.under_sampling.base import BaseCleaningSampler
import numpy as np
from copy import deepcopy

from rlearn.utils.validation import check_random_states
from itertools import product
from imblearn.pipeline import Pipeline

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

def check_pipelines(objects_list, random_state, n_runs):
    """
    TODO: check if random state generation is producing the expected outcomes
    """
    # Create random states
    random_states = check_random_states(random_state, n_runs)

    pipelines = []
    param_grid = []
    for comb in product(*objects_list):
        name  = '|'.join([i[0] for i in comb])
        comb = [(n,o,g) for n,o,g in comb if o is not None] # name, object, grid
        #names = [i[0] for i in comb]
        #objs  = [i[1] for i in comb]
        #grid  = [i[2] for i in comb]

        pipelines.append((name, Pipeline([(n,o) for n,o,g in comb])))

        grids = {'est_name': [name]}
        for obj_name, obj, sub_grid in comb:
            if 'random_state' in obj.get_params().keys():
                grids[f'{name}__{obj_name}__random_state'] = random_states
            for param, values in sub_grid.items():
                grids[f'{name}__{obj_name}__{param}'] = values
        param_grid.append(grids)

    return pipelines, param_grid
