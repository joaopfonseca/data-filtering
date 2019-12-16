import numpy as np
from gsmote import GeometricSMOTE
from sklearn.neighbors import NearestNeighbors

class DenoisedGeometricSMOTE(GeometricSMOTE):
    def __init__(self,
            sampling_strategy='auto',
            random_state=None,
            truncation_factor=1.0,
            deformation_factor=0.0,
            selection_strategy='combined',
            k_neighbors=5,
            k_neighbors_filter=3,
            n_jobs=1,):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            truncation_factor=truncation_factor,
            deformation_factor=deformation_factor,
            selection_strategy=selection_strategy,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs)
        self.k_neighbors_filter = k_neighbors_filter

    def _fit_resample(self, X, y):
        print('this?')
        _, indices = NearestNeighbors(n_neighbors=self.k_neighbors_filter, algorithm='auto')\
            .fit(X)\
            .kneighbors(X)
        print('not this')
        labels = np.vectorize(lambda x: y[x])(indices)
        status = np.equal(np.expand_dims(y,-1), labels).astype(int).sum(axis=1)/self.k_neighbors_filter>=0.5

        return super()._fit_resample(X[status], y[status])
