
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling.base import BaseCleaningSampler
from copy import deepcopy
from collections import Counter

class MBKMeansFiltering(BaseCleaningSampler):
    """My own method"""
    def __init__(self, filters, n_splits, granularity, method='obs_percent', threshold=0.7, random_state=None):
        assert method in ['obs_percent', 'mislabel_rate'], 'method must be either \'obs_percent\', \'mislabel_rate\''
        super().__init__(sampling_strategy='all')
        self.filters = deepcopy(filters)
        self.n_splits = n_splits
        self.granularity = granularity
        self.method = method
        self.threshold = threshold
        self.random_state = random_state

    def _fit_resample(self, X, y):
        #assert X.shape[0]==y.shape[0], 'X and y must have the same length.'
        ## cluster data
        index = np.arange(len(y))
        clusters_list = []
        index_list  = []
        self.kmeans = {}
        for analysis_label in np.unique(y):
            print(f'Label: {analysis_label}')
            label_indices = index[y==analysis_label]
            X_label = X[y==analysis_label]

            clusters, kmeans = self._KMeans_clustering(X_label)
            self.kmeans[analysis_label] = kmeans
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

        self.stratifiedkfold = StratifiedKFold(n_splits = self.n_splits, shuffle=True, random_state=self.random_state)
        self.filter_list = {}
        filter_outputs   = {}
        for n, (_, split) in enumerate(self.stratifiedkfold.split(X, y_)):
            print(f'Applying filter {n}')
            for name, clf in self.filters:
                classifier = deepcopy(clf)
                classifier.fit(X[split], y_[split])
                filter_outputs[f'filter_{n}_{name}'] = classifier.predict(X)
                self.filter_list[f'{n}_{name}'] = classifier
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

    def fit(self, X, y):
        """Fits filter to X, y."""
        self._fit_resample(X, y)
        return self

    def resample(self, X, y):
        print('Resampling... (check if this happens when using predict method on a pipeline, it shouldn\'t)')
        index = np.arange(len(y))
        clusters_list = []
        index_list  = []
        for analysis_label in np.unique(y):
            print(f'Label: {analysis_label}')
            label_indices = index[y==analysis_label]
            X_label = X[y==analysis_label]

            clusters = self.kmeans[analysis_label].predict(X_label)
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

        filter_outputs   = {}
        for name, classifier in self.filter_list.items():
            filter_outputs[f'filter_{name}'] = classifier.predict(X)
            print(f'Applied classifier {name.split("_")[-1]} (part of filter {name.split("_")[0]})')

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

class EnsembleFiltering(BaseCleaningSampler):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, filters, n_splits=4, threshold=0.5, random_state=None):
        super().__init__(sampling_strategy='all')
        self.filters = deepcopy(filters)
        self.n_splits = n_splits
        self.threshold = threshold
        self.random_state = random_state

    def _fit_resample(self, X, y):

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        self.filter_list = {}
        filter_outputs = {f'filter_{name}':np.zeros((y.shape))-1 for name, _ in self.filters}
        self.stratifiedkfold = StratifiedKFold(n_splits = self.n_splits, shuffle=True, random_state=random_state)
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            print(f'Applying filter {n}')
            for name, clf in self.filters:
                classifier = deepcopy(clf)
                classifier.fit(X[train_indices], y_[train_indices])
                filter_outputs[f'filter_{name}'][test_indices] = classifier.predict(X[test_indices])
                self.filter_list[f'{n}_{name}'] = classifier
                print(f'Applied classifier {name} (part of filter {n})')
        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters
        ## filter data
        self.status = mislabel_rate<=self.threshold
        return X[self.status], y[self.status]

    def fit(self, X, y):
        self._fit_resample(X, y)
        return self

    def resample(self, X, y):

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        indices = []
        filter_outputs = {f'filter_{name}':np.zeros((y.shape))-1 for name, _ in self.filters}
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            print(f'Applying filter {n}')
            for name in dict(self.filters).keys():
                filter_outputs[name][test_indices] = self.filter_list[f'{n}_{name}'].predict(X[test_indices])

                print(f'Applied classifier {name} (part of filter {n})')
        ## mislabel rate
        total_filters = len(filter_outputs.keys())
        mislabel_rate = (total_filters - \
            np.apply_along_axis(
                lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                .astype(int).sum(axis=1)
                )/total_filters
        ## filter data
        self.status = mislabel_rate<=self.threshold
        return X[self.status], y[self.status]

class CompositeFilter(BaseCleaningSampler):
    """
    Based on "Novel mislabeled training data detection algorithm",
    Yuan, Guan, Zhu et al. (2018).
    `method`: `MFMF`, `CFCF`, `CFMF`, `MFCF`
    """
    def __init__(self, filters, method='MFMF', n_splits=4, random_state=None):
        assert  len(method)==4\
            and method[-2:] in ['MF', 'CF']\
            and method[-2:] in ['MF', 'CF'], \
            'Invalid `method` value passed.'

        super().__init__(sampling_strategy='all')
        self.filters = filters
        self.method = method

    def _fit_resample(self, X, y):
        if self.method.startswith('MF'): self.threshold_1 = .5
        else: self.threshold_1 = 1-.9e-15

        if self.method.endsswith('MF'): self.threshold_2 = .5
        else: self.threshold_2 = 1-.9e-15

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        self.filter_list = {}
        voted_outputs_1 = {}
        indices = []
        self.stratifiedkfold = StratifiedKFold(n_splits = self.n_splits, shuffle=True, random_state=random_state)
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            print(f'Applying filter {n}')
            filter_outputs = {}
            for name, clf in self.filters:
                classifier = deepcopy(clf)
                classifier.fit(X[train_indices], y_[train_indices])
                filter_outputs[f'filter_{name}'] = classifier.predict(X)
                self.filter_list[f'{n}_{name}'] = classifier
                print(f'Applied classifier {name} (part of filter {n})')
            total_filters = len(filter_outputs.keys())
            voted_outputs_1[n] = ((total_filters - \
                np.apply_along_axis(
                    lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                    .astype(int).sum(axis=1)
                    )/total_filters) <= self.threshold_1

        ## mislabel rate
        total_votes = len(voted_outputs_1.keys())
        mislabel_rate = (pd.DataFrame(voted_outputs_1).values\
                .astype(int).sum(axis=1))/total_votes
        ## filter data
        self.status = mislabel_rate<=self.threshold_2
        return X[self.status], y[self.status]

    def fit(self, X, y):
        self._fit_resample(X, y)
        return self

    def resample(self, X, y):
        if self.method.startswith('MF'): self.threshold_1 = .5
        else: self.threshold_1 = 1-.9e-15

        if self.method.endsswith('MF'): self.threshold_2 = .5
        else: self.threshold_2 = 1-.9e-15

        label_encoder = LabelEncoder()
        y_ = label_encoder.fit_transform(y)

        ## run filter
        voted_outputs_1 = {}
        for n, (train_indices, test_indices) in enumerate(self.stratifiedkfold.split(X, y_)):
            print(f'Applying filter {n}')
            filter_outputs = {}
            for name, clf in self.filters:
                filter_outputs[f'filter_{name}'] = self.filter_list[f'{n}_{name}'].predict(X)

            total_filters = len(filter_outputs.keys())
            voted_outputs_1[n] = ((total_filters - \
                np.apply_along_axis(
                    lambda x: x==y_, 0, pd.DataFrame(filter_outputs).values)\
                    .astype(int).sum(axis=1)
                    )/total_filters) <= self.threshold_1

        ## mislabel rate
        total_votes = len(voted_outputs_1.keys())
        mislabel_rate = (pd.DataFrame(voted_outputs_1).values\
                .astype(int).sum(axis=1))/total_votes
        ## filter data
        self.status = mislabel_rate<=self.threshold_2
        return X[self.status], y[self.status]

class ConsensusFiltering(EnsembleFiltering):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, filters, n_splits=4, random_state=None):
        super().__init__(filters, n_splits=n_splits, threshold=1-.9e-15, random_state=random_state)

class MajorityVoteFiltering(EnsembleFiltering):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, filters, n_splits=4, random_state=None):
        super().__init__(filters, n_splits=n_splits, threshold=.5, random_state=random_state)

class SingleFilter(EnsembleFiltering):
    """Identifying Mislabeled Training Data, by Brodley and Friedl (1999)"""
    def __init__(self, filter, n_splits=4):
        filters = [(filter.__class__, filter)]
        super().__init__(filters, n_splits=n_splits, threshold=.5, random_state=random_state)

class ChainFiltering(BaseCleaningSampler):
    """Own method"""
    def __init__(self, filter_obj, stopping_criteria='manual', tol=None, max_iter=40, random_state=None):
        assert stopping_criteria in ['auto', 'manual'],  '`stopping_criteria` must be either `auto` or `manual`'
        assert tol if stopping_criteria=='auto', '`tol` must be defined while `stopping_criteria` is defined as `auto`'
        self.filter_methods = [filter_obj for _ in range(max_iter)]
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter

    def _fit_resample(self, X, y):
        X_nnf, y_nnf = X.copy(), y.copy()
        self.filter_list = {}
        for n, filter in enumerate(self.filter_methods):
            filter = filter.fit(X_nnf, y_nnf)
            X_nnf, y_nnf = filter.resample(X, y)
            self.filter_list[n] = filter
            if n!=0 and method='auto':
                not_changed = dict(Counter(self.filter_list[n-1].status == self.filter_list[n].status))
                percent_changes = not_changed[False]/sum(not_changed.values())
                print(f'Percentage of status changes: {percent_changes*100}%')
                if percent_changes<=tol:
                    break

        self.final_filter = filter
        return X_nnf, y_nnf



#class PartitionedFiltering:
#    """
#    Based on "Novel mislabeled training data detection algorithm",
#    Yuan, Guan, Zhu et al. (2018)
#    """

class YuanGuanZhu(ChainFiltering):
    """
    Novel mislabeled training data detection algorithm, Yuan, Guan, Zhu et al. (2018)
    Filters used in paper: naive Bayes, decision tree, and 3-Nearest Neighbor
    """
    def __init__(self, filters, n_splits=3, t=40, method='majority', random_state=None):
        """method: `majority` or `consensus`"""
        assert method in ['majority', 'consensus'], '`method` must be either `majority` or `minority`.'
        filter_obj = [MajorityVoteFiltering() if method=='majority' else ConsensusFiltering()]
        super().__init__()
        self.method = method

    def _fit_resample(self, X, y):
        pass
