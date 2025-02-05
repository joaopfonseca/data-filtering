import logging
import click
import os
import pickle
import numpy as np
import pandas as pd
from itertools import product
from src.models.data_selection import (
    MBKMeansFilter,
    SingleFilter,
    ConsensusFilter,
    MajorityVoteFilter,
    CompositeFilter,
    YuanGuanZhu,
    ChainFilter,
    MBKMeansFilter_reversed
)

from src.data.make_dataset import importdb
from src.reporting.reports import reports
from src.experiment.utils import make_multiclass_noise, check_pipelines
from src.models.oversampling import DenoisedGeometricSMOTE

#from rlearn.tools.experiment import ImbalancedExperiment
from rlearn.utils.validation import check_oversamplers_classifiers
from rlearn.model_selection import ModelSearchCV
from rlearn.tools.reporting import report_model_search_results
from rlearn.utils.validation import check_random_states

from imblearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

## read data
df = pd.read_csv('data/raw/lucas.csv')
X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
X = X[~np.isin(y, [5,6,7])]
y = y[~np.isin(y, [5,6,7])]
transfer_map = {0:3,1:2,2:4,3:0,4:0}
#labels = {'A':1, 'B':2, 'C':0, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}

## preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)


random_state = 0

## classifiers for filters
filts = (
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=25, random_state=random_state)),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state, multi_class='auto', max_iter=750)),
    ('MLPClassifier', MLPClassifier(random_state=random_state, max_iter=2000))
)
single_filter = RandomForestClassifier(n_estimators=25, random_state=random_state)

## introduce noise
noise_objs = [
    ('no_noise', None, {}),
    ('noise5', make_multiclass_noise(transfer_map, noise_level=.05), {}),
    ('noise10', make_multiclass_noise(transfer_map, noise_level=.1), {}),
    ('noise20', make_multiclass_noise(transfer_map, noise_level=.2), {}),
    ('noise30', make_multiclass_noise(transfer_map, noise_level=.3), {}),
    ('noise40', make_multiclass_noise(transfer_map, noise_level=.4), {})
]

## filter data
data_filters = [
    ('no_filter', None, {}),
    ('singlefilter', SingleFilter(), {'n_splits':[3,4,5,6,7,8]}),
    ('consensusfilter', ConsensusFilter(), {'n_splits':[3,4,5,6,7,8]}),
    ('majorityfilter', MajorityVoteFilter(), {'n_splits':[3,4,5,6,7,8]}),
    ('mymethod', MBKMeansFilter(),
            {
            'n_splits':[3,4,5,6,7], 'granularity':[.1,.5,1,3,4,5],
            'method':['obs_percent', 'mislabel_rate'],
            'threshold':[.25, .5, .75, .99]
            })
    ('mymethod_reversed', MBKMeansFilter_reversed(),
            {
            'n_splits':[2,3,4,5,6,7], 'granularity':[.5,1,1.5,2,3,4,5],
            'method':['obs_percent', 'mislabel_rate'],
            'threshold':[.25, .5, .75, .99]
            }),
]

oversamplers = [
#    ('None', None, {}),
    ('DenoisedGeometricSMOTE', DenoisedGeometricSMOTE(),
            {
            'k_neighbors': [3],
            'selection_strategy': ['combined', 'minority', 'majority'],
            'truncation_factor': [-.5,0,.5],
            'deformation_factor': [0,.5,1],
            'k_neighbors_filter': [3,5]
            })
]

classifiers = [
    ('randomforestclassifier', RandomForestClassifier(n_estimators=100, random_state=random_state), {})
]



objects_list = [noise_objs, data_filters, oversamplers, classifiers]
pipelines, param_grid = check_pipelines(objects_list, random_state, 1)


fit_params = {}
for clf_name in list(dict(pipelines).keys()):
    clf_name_split = clf_name.split('|')
    if clf_name_split[1]=='DenoisedGeometricSMOTE':
        pass
    elif clf_name_split[1]=='singlefilter':
        fit_params[f'{clf_name_split[1]}__filters'] = [single_filter]
    elif clf_name_split[1]!='no_filter':
        fit_params[f'{clf_name_split[1]}__filters'] = filts

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
model_search = ModelSearchCV(pipelines, param_grid, n_jobs=-1, cv=cv, verbose=1)
model_search.fit(X,y,**fit_params)
df_results = report_model_search_results(model_search)\
    .sort_values('mean_test_score', ascending=False)
df_results.to_csv('results.csv')
pickle.dump(model_search, open('./model_search.pkl','wb'))
