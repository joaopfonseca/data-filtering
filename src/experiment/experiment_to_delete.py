import logging
import click
import os
import pickle
import numpy as np
from ..models.data_selection import KMeans_filtering
from ..data.make_dataset import importdb
from ..reporting.reports import reports

from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

input_filepath='data/interim/pixel_selection.pkl'; noise_levels=[0.2]; random_state=None
filters = (
    ('RandomForestClassifier', RandomForestClassifier(random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(random_state=random_state)),
    ('MLPClassifier', MLPClassifier(random_state=random_state))
)
n_splits=5; granularity=5; threshold=0.7; method='mislabel_rate' # 'mislabel_rate'


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.25, random_state=42)

X_train_original = X_train.copy()
y_train_original = y_train.copy()

mask = np.array(
        [1 for i in range(int(len(y_train)*noise_level))] + \
        [0 for i in range(len(y_train)-int(len(y_train)*noise_level))]
    ).astype(bool)
np.random.RandomState(random_state)
np.random.shuffle(mask)
y_train[mask] = np.vectorize(lambda x: 0 if x==1 else 1)(y_train[mask])
## run data selection
kmf = KMeans_filtering(filters, n_splits, granularity,
method, threshold=0.7, random_state=None)
rfc = RandomForestClassifier()
clf = make_pipeline(kmf, rfc)
clf.fit(X_train, y_train)

## train baseline
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

## train
rfc_no_noise = RandomForestClassifier()
rfc_no_noise.fit(X_train_original, y_train_original)

## data selection quality analysis
labels = {0: 'Correctly labelled', 1:'Noise'}
selection_reports = reports(mask, ~dict(clf.steps)['kmeans_filtering'].status, labels)

## test classification quality
clf_ds_reports = reports(y_test, clf.predict(X_test), {0:0,1:1})
baseline_reports = reports(y_test, rfc.predict(X_test), {0:0,1:1})
no_noise_reports = reports(y_test, rfc_no_noise.predict(X_test), {0:0,1:1})
print('noise 20% + data selection + rfc\n',clf_ds_reports[-1])
print('\n\nnoise 20% + rfc\n',baseline_reports[-1])
print('\n\nno noise + rfc\n',no_noise_reports[-1])


def mknoise_and_filter(
            X, y, noise_level,
            method, threshold, filters, n_splits, granularity,
            random_state=None
            ):
        ## introduce noise
        mask = np.array(
            [1 for i in range(int(len(y)*noise_level))] + \
            [0 for i in range(len(y)-int(len(y)*noise_level))]
        ).astype(bool)
        np.random.RandomState(random_state)
        np.random.shuffle(mask)
        y[mask] = np.vectorize(lambda x: 0 if x==1 else 1)(y[mask])
        ## run data selection
        kmf = KMeans_filtering(filters, n_splits, granularity,
            method, threshold=0.7, random_state=None)
        rfc = RandomForestClassifier()
        clf = make_pipeline(kmf, rfc)
        clf.fit(X,y)

        ## save quality analysis of data selection
        labels = {0: 'Correctly labelled', 1:'Noise'}
        selection_reports = reports(mask, ~dict(clf.steps)['kmeans_filtering'].status, labels)
        ## filter data
        X_filtered = X[selection_results]
        y_filtered = y[selection_results]
        ## train
        pass
        ## report


def main(   input_filepath, method, noise_levels,
            filters, n_splits, granularity, threshold,
            random_state=None
            ):
    ## read data
    if input_filepath.split('.')[-1]=='pkl':
        data = pickle.load(open(input_filepath, 'rb'))
    elif input_filepath.split('.')[-1]=='db':
        data = importdb(input_filepath)
    elif os.path.isfile(input_filepath):
        raise IOError('File is neither a Pickle dictionary nor SQLite3 database.')
    else:
        raise IOError('input_filepath not recognized as file.')

    results = {}
    for noise_level in noise_levels:
        for name, data_dict in data.items():
            X = data_dict['X']
            y = data_dict['y']

            ## introduce noise
            mask = np.array(
                [1 for i in range(int(len(y)*noise_level))] + \
                [0 for i in range(len(y)-int(len(y)*noise_level))]
            ).astype(bool)
            np.random.RandomState(random_state)
            np.random.shuffle(mask)
            y[mask] = np.vectorize(lambda x: 0 if x==1 else 1)(y[mask])
            ## run data selection
            selection_results = KMeans_filtering(
                X, y, filters, n_splits, granularity,
                method, mislabel_rate_threshold, random_state=random_state)[-1]
            ## save quality analysis of data selection
            labels = {0: 'Correctly labelled', 1:'Noise'}
            selection_reports = reports(mask, ~selection_results, labels)
            ## filter data
            X_filtered = X[selection_results]
            y_filtered = y[selection_results]
            ## train
            pass
            ## report






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)