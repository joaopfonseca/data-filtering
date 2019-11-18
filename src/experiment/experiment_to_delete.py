import logging
import click
import os
import pickle
import numpy as np
from ..models.data_selection import (
    MBKMeansFilter,
    SingleFilter,
    ConsensusFilter,
    MajorityVoteFilter,
    YuanGuanZhu,
    ChainFilter
)

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
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state)),
    ('MLPClassifier', MLPClassifier(random_state=random_state))
)
n_splits=5; granularity=5; threshold=0.7; method='mislabel_rate' # 'mislabel_rate'
noise_level = 0.2



X, y = pickle.load(open(input_filepath, 'rb'))['YEAST 1 (3)'].values()

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

all_reports = {}
selection_reports = {}

## train baseline
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
all_reports['rfc_noise'] = reports(y_test, rfc.predict(X_test), {0:0,1:1})

## train no noise
rfc_no_noise = RandomForestClassifier()
rfc_no_noise.fit(X_train_original, y_train_original)
all_reports['rfc_no_noise'] = reports(y_test, rfc_no_noise.predict(X_test), {0:0,1:1})

## run data selection
# MBKMeansFilter, SingleFilter, ConsensusFilter, MajorityVoteFilter, YuanGuanZhu, ChainFilter
labels = {0: 'Correctly labelled', 1:'Noise'}
# MBKMeansFilter
kmf = MBKMeansFilter(filters, n_splits, granularity, method, threshold=0.7, random_state=None)
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
