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
from rlearn.model_selection import ModelSearchCV
from rlearn.tools.reporting import report_model_search_results

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
df = pd.read_csv('data/raw/dgt_preprocessed_data.csv')
df = df.dropna()

X, y, obj = df.drop(columns=['X','Y','Object', 'Label']).values, df.Label.values, df.Object.values

## Encode Labels
label_map = {k:v for k, v in zip(range(len(np.unique(y))), np.unique(y))}
y = np.fromiter(map(lambda x: {v:k for k,v in label_map.items()}[x], y), dtype=int)
## preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)
## generate transfer map
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
cm = reports(y_test, y_pred, {i:i for i in range(len(label_map))})[1]
cmun = cm.unstack().reset_index()
cmun = cmun[
    ~cmun[['level_0', 'level_1']].isin(['UA', 'PA','Total']).values.any(axis=1)
]
cmun = cmun[
    cmun['level_0']!=cmun['level_1']
]
cmun[0] = cmun[0].apply(lambda x: x.replace(',', '')).astype(int)
cmun = cmun.sort_values(0, ascending=False).drop_duplicates(['level_0'], keep='first')
transfer_map = {k:v for k,v in zip(cmun['level_0'], cmun['level_1'])}

random_state = 0

## classifiers for filters
filts = (
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=25, random_state=random_state)),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state, multi_class='auto', max_iter=750)),
    ('MLPClassifier', MLPClassifier(random_state=random_state, max_iter=2000))
)
single_filter = DecisionTreeClassifier(random_state=random_state)

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
            }),
    ('mymethod_reversed', MBKMeansFilter_reversed(),
            {
            'n_splits':[2,3,4,5,6,7], 'granularity':[.5,1,1.5,2,3,4,5],
            'method':['obs_percent', 'mislabel_rate'],
            'threshold':[.25, .5, .75, .99]
            }),
]

#oversamplers = [
##    ('None', None, {}),
#    ('DenoisedGeometricSMOTE', DenoisedGeometricSMOTE(),
#            {
#            'k_neighbors': [3],
#            'selection_strategy': ['combined', 'minority', 'majority'],
#            'truncation_factor': [-.5,0,.5],
#            'deformation_factor': [0,.5,1],
#            'k_neighbors_filter': [3,5]
#            })
#]

classifiers = [
    ('randomforestclassifier', RandomForestClassifier(n_estimators=100, random_state=random_state), {})
]



objects_list = [noise_objs, data_filters, #oversamplers,
    classifiers]
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
