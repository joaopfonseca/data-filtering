import logging
import click
import os
import pickle
import numpy as np
from src.models.data_selection import (
    MBKMeansFilter,
    SingleFilter,
    ConsensusFilter,
    MajorityVoteFilter,
    YuanGuanZhu,
    ChainFilter
)

from src.data.make_dataset import importdb
from src.reporting.reports import reports
from src.experiment.utils import make_noise

from rlearn.tools.experiment import ImbalancedExperiment

from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

input_filepath='data/interim/pixel_selection.pkl'; noise_levels=[0.2]; random_state=123
filters = (
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=25, random_state=random_state)),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state)),
    ('MLPClassifier', MLPClassifier(random_state=random_state))
)

n_splits=7; granularity=3; threshold=0.5; method='mislabel_rate' # 'mislabel_rate'
noise_level = 0.2

## save data objects
all_reports = {}
selection_reports = {}


## read data
data = pickle.load(open(input_filepath, 'rb'))
for dname in data.items()


## introduce noise

noise_objs = [
    (f'{noise_percent}_noise', make_noise(noise_percent, random_state)) for noise_percent in [.05, .1, .2, .3, .4]
] + [('none', None)]

data_filters = [
    ('none', None),
    ('singlefilter', SingleFilter(filters[0][1])),
    ('consensusfilter', ConsensusFilter(filters)),
    ('majorityfilter', MajorityVoteFilter(filters)),
    ('mymethod', MBKMeansFilter(filters)),
    ('yuanguanzhu', YuanGuanZhu(filters))
]

classifiers = [
    ('randomforestclassifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
]



for name, noisify in noise_objs:
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.25, random_state=random_state)


ImbalancedExperiment()

## train baseline
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_noisy, y_train)
all_reports['rfc_noise'] = reports(y_test, rfc.predict(X_test), {0:0,1:1})

## train no noise
rfc_no_noise = RandomForestClassifier(n_estimators=100)
rfc_no_noise.fit(X_train_original, y_train_original)
all_reports['rfc_no_noise'] = reports(y_test, rfc_no_noise.predict(X_test), {0:0,1:1})

## run data selection
# MBKMeansFilter, SingleFilter, ConsensusFilter, MajorityVoteFilter, YuanGuanZhu, ChainFilter
labels = {0: 'Correctly labelled', 1:'Noise'}
# MBKMeansFilter
kmf = MBKMeansFilter(filters, 4, 5, method='mislabel_rate', threshold=0.7, random_state=None)
rfc = RandomForestClassifier(n_estimators=100)
clf = make_pipeline(kmf, rfc)
clf.fit(X_train, y_train)
all_reports['MBKMeansFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
selection_reports['MBKMeansFilter'] = reports(mask, ~dict(clf.steps)['MBKMeansFilter'.lower()].status, labels)

# SingleFilter
sf = SingleFilter(filters[0][-1], n_splits=4)
rfc = RandomForestClassifier(n_estimators=100)
clf = make_pipeline(sf, rfc)
clf.fit(X_train, y_train)
all_reports['SingleFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
selection_reports['SingleFilter'] = reports(mask, ~dict(clf.steps)['SingleFilter'.lower()].status, labels)

# ConsensusFilter
CF = ConsensusFilter(filters)
rfc = RandomForestClassifier(n_estimators=100)
clf = make_pipeline(CF, rfc)
clf.fit(X_train, y_train)
all_reports['ConsensusFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
selection_reports['ConsensusFilter'] = reports(mask, ~dict(clf.steps)['ConsensusFilter'.lower()].status, labels)

# MajorityVoteFilter
MF = MajorityVoteFilter(filters)
rfc = RandomForestClassifier(n_estimators=100)
clf = make_pipeline(MF, rfc)
clf.fit(X_train, y_train)
all_reports['MajorityVoteFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
selection_reports['MajorityVoteFilter'] = reports(mask, ~dict(clf.steps)['MajorityVoteFilter'.lower()].status, labels)

# YuanGuanZhu Majority
YGZM = YuanGuanZhu(filters, method='majority')
rfc = RandomForestClassifier(n_estimators=100)
clf = make_pipeline(YGZM, rfc)
clf.fit(X_train, y_train)
all_reports['YuanGuanZhu Majority'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
selection_reports['YuanGuanZhu Majority'] = reports(mask, ~dict(clf.steps)['YuanGuanZhu'.lower()].status, labels)

# YuanGuanZhu Consensus
YGZC = YuanGuanZhu(filters, method='consensus')
rfc = RandomForestClassifier(n_estimators=100)
clf = make_pipeline(YGZC, rfc)
clf.fit(X_train, y_train)
all_reports['YuanGuanZhu Consensus'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
selection_reports['YuanGuanZhu Consensus'] = reports(mask, ~dict(clf.steps)['YuanGuanZhu'.lower()].status, labels)


scores = {}
for name, reps in all_reports.items():
    scores[name] = reps[-1]

pd.concat(scores).reset_index().pivot('level_0','level_1','Score')
