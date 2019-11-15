
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling.base import BaseCleaningSampler
from copy import deepcopy

class KMeans_filtering(BaseCleaningSampler):
    def __init__(self, filters, n_splits, granularity, method='obs_percent', threshold=0.7, random_state=None):
        assert method in ['obs_percent', 'mislabel_rate'], 'method must be either \'obs_percent\', \'mislabel_rate\''
        super().__init__(sampling_strategy='all')
        self.filters = filters
        self.n_splits = n_splits
        self.granularity = granularity
        self.method = method
        self.threshold = threshold
        self.random_state = random_state

    def _fit_resample(self, X, y):
        assert X.shape[0]==y.shape[0], 'X and y must have the same length.'

        ## cluster data
        index = np.arange(len(y))
        clusters_list = []
        index_list  = []
        for analysis_label in np.unique(y):
            print(f'Label: {analysis_label}')
            label_indices = index[y==analysis_label]
            X_label = X[y==analysis_label]

            clusters, kmeans = self._KMeans_clustering(X_label)
            index_list.append(label_indices)
            clusters_list.append(clusters)

        ## cluster labels
        cluster_col = pd.Series(
            data=np.concatenate(clusters_list),
            index=np.concatenate(index_list),
            name='cluster')\
            .sort_index()

        ## apply filters
        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=random_state)
        splits = []
        for _, split_indices in skf.split(X, y_):
            splits.append(split_indices)

        filter_outputs = {}
        for n, split in enumerate(splits):
            print(f'Applying filter {n}')
            for name, clf in filters:
                classifier = deepcopy(clf)
                classifier.fit(X[split], y_[split])
                filter_outputs[f'filter_{n}_{name}'] = classifier.predict(X)
                print(f'Applied classifier {name} (part of filter {n})')

        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters

        ## crunch data
        mislabel_col = pd.Series(data=mislabel_rate, index=index, name='mislabel_rate')
        y_col = pd.Series(data=y, index=index, name='y')
        df = cluster_col.to_frame().join(y_col).join(mislabel_col)
        df['count'] = 1
        df_cluster_info_grouped = df.groupby(['y', 'cluster'])\
                    .agg({'mislabel_rate':np.mean, 'count':'count'})\
                    .sort_values(['mislabel_rate'])
        df_cluster_info_A = df_cluster_info_grouped.groupby(['y']).cumsum()\
            .rename(columns={'count':'cumsum'}).drop(columns=['mislabel_rate'])
        df_cluster_info = df_cluster_info_grouped.join(df_cluster_info_A)

        if self.method=='mislabel_rate':
            df_cluster_info['status'] = df_cluster_info['mislabel_rate']<=self.threshold
        elif self.method=='obs_percent':
            thresholds = df_cluster_info.groupby('y').max()['cumsum']*self.threshold
            actual_thresholds = df_cluster_info[
                    df_cluster_info['cumsum']/thresholds>=1
                ]['cumsum'].groupby('y').min()
            df_cluster_info['status'] = df_cluster_info['cumsum']/actual_thresholds<=1

        results = df.join(df_cluster_info['status'], on=['y','cluster'])
        self.status = results['status'].values
        return X[self.status], y[self.status]

    def _KMeans_clustering(self, X):
        """Private function to..."""
        if self.granularity>=np.sqrt(X.shape[0]):
            self.granularity=int(np.sqrt(X.shape[0]))-1
            print(f'Granularity too high for passed array, clipping to {self.granularity}')
        k = int(self.granularity*np.sqrt(X.shape[0]))
        kmeans = MiniBatchKMeans(k, max_iter=5*k, tol=0, max_no_improvement=400, random_state=self.random_state)
        labels = kmeans.fit_predict(X).astype(str)
        return labels, kmeans






def brodley_friedl_method(X, y, filters, n_splits):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    filter_outputs = {}
    for n, split in enumerate(splits):
        print(f'Applying filter {n}')
        for name, clf in filters:
            classifier = deepcopy(clf)
            classifier.fit(X[split], y_[split])
            filter_outputs[f'filter_{n}_{name}'] = classifier.predict(X)
            print(f'Applied classifier {name} (part of filter {n})')

    ## mislabel rate
    total_filters = len(filter_outputs.keys())
    mislabel_rate = (total_filters - \
        np.apply_along_axis(
            lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
            .astype(int).sum(axis=1)
            )/total_filters
