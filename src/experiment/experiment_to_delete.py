import logging
import click
import os
import pickle
import numpy as np
import pandas as pd
from src.models.data_selection import (
    MBKMeansFilter,
    SingleFilter,
    ConsensusFilter,
    MajorityVoteFilter,
    CompositeFilter,
    YuanGuanZhu,
    ChainFilter
)

from src.data.make_dataset import importdb
from src.reporting.reports import reports
from src.experiment.utils import make_binary_noise

#from rlearn.tools.experiment import ImbalancedExperiment
from rlearn.utils.validation import check_oversamplers_classifiers
from rlearn.model_selection import ModelSearchCV
from rlearn.tools.reporting import report_model_search_results

from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

input_filepath='data/interim/pixel_selection.pkl'; random_state=12
filts = (
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=25, random_state=random_state)),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, random_state=random_state)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),
    ('LogisticRegression', LogisticRegression(solver='lbfgs', random_state=random_state)),
    ('MLPClassifier', MLPClassifier(random_state=random_state))
)
single_filter = RandomForestClassifier(n_estimators=25, random_state=random_state)

n_splits=4; granularity=3; threshold=0.5; method='mislabel_rate' # 'mislabel_rate'
noise_level = 0.2

## save data objects
all_reports = {}
selection_reports = {}


## read data
data = pickle.load(open(input_filepath, 'rb'))

## introduce noise
noise_objs = [
    ('no_noise', None, {}),
    ('noise5', make_binary_noise(noise_level=.05), {}),#{'noise_level':[.05, .1, .2, .3, .4]})
    ('noise20', make_binary_noise(noise_level=.2), {})
]

#data_filters = [
#    ('no_filter', None, {}),
#    ('singlefilter', SingleFilter(), {'n_splits':[3,4,5,6,7,8]}),
#    ('consensusfilter', ConsensusFilter(), {'n_splits':[3,4,5,6,7,8]}),
#    ('majorityfilter', MajorityVoteFilter(), {'n_splits':[3,4,5,6,7,8]}),
#    ('mymethod', MBKMeansFilter(),
#            {
#            'n_splits':[3,4,5,6,7,8], 'granularity':[3,5,9],
#            'method':['obs_percent', 'mislabel_rate'],
#            'threshold':[.1, .25, .5, .75, .99]
#            })
#]

data_filters = [
    ('no_filter', None, {}),
    ('singlefilter', SingleFilter(), {}),
    ('consensusfilter', ConsensusFilter(), {}),
    ('majorityfilter', MajorityVoteFilter(), {}),
    ('mymethod', MBKMeansFilter(granularity=7, method='mislabel_rate',threshold=.99),
            {})
]


classifiers = [
    ('randomforestclassifier', RandomForestClassifier(n_estimators=100, random_state=random_state), {})
]

prepre_pipelines, pre_param_grid = check_oversamplers_classifiers(data_filters, classifiers, random_state, 1)
pre_pipelines = pd.DataFrame(prepre_pipelines).set_index(0)

pre_param_grid2 = pd.DataFrame(pre_param_grid)
pre_param_grid2['est_name'] = pre_param_grid2['est_name'].apply(lambda x: x[0])

all_classifiers = []
for est in pd.DataFrame(pre_param_grid)['est_name'].apply(lambda x: x[0]).unique():
    _pre_params = pre_param_grid2.set_index('est_name')\
        .loc[est, [x.startswith(est) for x in pre_param_grid2.set_index('est_name').columns]]

    if type(_pre_params) == pd.Series:
        _pre_params = _pre_params.apply(lambda x: x[0]).to_dict()
    else:
        _pre_params = _pre_params.apply(lambda x: x.apply(lambda x: x[0]), axis=1)\
        .groupby('est_name').agg(list).to_dict(orient='list')

    pre_params = {}
    for key, values in _pre_params.items():
        pre_params[key] = list(np.unique(values))

    all_classifiers.append((est, pre_pipelines.loc[est,1], pre_params))


pipelines, param_grid = check_oversamplers_classifiers(noise_objs, all_classifiers, random_state, 1)

new_pipelines = {k:[(k,v)] for k,v in pipelines}
new_param_grid = {}
for dictionary in param_grid:
    new_param_grid[dictionary['est_name'][0]] = []

for dictionary in param_grid:
    new_dictionary = {}
    for k, v in dictionary.items():
        if k=='est_name':
            pass
        elif '|' in k.split('__')[1]:
            new_k = k.split('__')
            new_k.pop(1)
            k = '__'.join(new_k)

        new_dictionary[k] = v
    new_param_grid[new_dictionary['est_name'][0]].append(new_dictionary)


# temp
df = pd.read_csv('../publications/remote-sensing-lucas/data/lucas.csv')
X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
#X, y = pickle.load(open(input_filepath, 'rb'))['YEAST 1 (3)'].values()

cv_results = {}
for clf_name in list(dict(pipelines).keys()):
    fit_params = {}
    clf_name_split = clf_name.split('|')
    if clf_name_split[1]=='singlefilter':
        fit_params[f'{clf_name_split[1]}__filters'] = [single_filter]
    elif clf_name_split[1]!='no_filter':
        fit_params[f'{clf_name_split[1]}__filters'] = filts

    model_search = ModelSearchCV(new_pipelines[clf_name], new_param_grid[clf_name], n_jobs=-1, cv=5, verbose=1)
    model_search.fit(X,y, **fit_params)
    cv_results[clf_name] = model_search

all_results = {}
for exp_name, results in cv_results.items():
    all_results[exp_name] = report_model_search_results(results)

results = pd.concat(all_results.values())
results.sort_values('mean_test_score', ascending=False)






### pseudo-replicate experiment
kmf = MBKMeansFilter(granularity=7, method='obs_percent', threshold=1, random_state=662124363)
noise5 = make_binary_noise(noise_level=.05, random_state=662124363)
rfc = RandomForestClassifier(n_estimators=100, random_state=662124363)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_noisy, y_noisy = noise5.fit_resample(X_train, y_train)
X_filtered, y_filtered = kmf.fit_resample(X_noisy, y_noisy, filts)
rfc.fit(X_filtered, y_filtered)
y_pred = rfc.predict(X_test)
reports(y_test, y_pred, {k:k for k in np.unique(y)})







### train baseline
#rfc = RandomForestClassifier(n_estimators=100)
#rfc.fit(X_noisy, y_train)
#all_reports['rfc_noise'] = reports(y_test, rfc.predict(X_test), {0:0,1:1})
#
### train no noise
#rfc_no_noise = RandomForestClassifier(n_estimators=100)
#rfc_no_noise.fit(X_train_original, y_train_original)
#all_reports['rfc_no_noise'] = reports(y_test, rfc_no_noise.predict(X_test), {0:0,1:1})
#
### run data selection
## MBKMeansFilter, SingleFilter, ConsensusFilter, MajorityVoteFilter, YuanGuanZhu, ChainFilter
#labels = {0: 'Correctly labelled', 1:'Noise'}
## MBKMeansFilter
#kmf = MBKMeansFilter(4, 5, method='mislabel_rate', threshold=0.7, random_state=None)
#rfc = RandomForestClassifier(n_estimators=100)
#clf = make_pipeline(kmf, rfc)
#clf.fit(X_train, y_train, **{'mbkmeansfilter__filters':filters})
#all_reports['MBKMeansFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
#selection_reports['MBKMeansFilter'] = reports(mask, ~dict(clf.steps)['MBKMeansFilter'.lower()].status, labels)
#
## SingleFilter
#sf = SingleFilter(filters[0][-1], n_splits=4)
#rfc = RandomForestClassifier(n_estimators=100)
#clf = make_pipeline(sf, rfc)
#clf.fit(X_train, y_train)
#all_reports['SingleFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
#selection_reports['SingleFilter'] = reports(mask, ~dict(clf.steps)['SingleFilter'.lower()].status, labels)
#
## ConsensusFilter
#CF = ConsensusFilter(filters)
#rfc = RandomForestClassifier(n_estimators=100)
#clf = make_pipeline(CF, rfc)
#clf.fit(X_train, y_train)
#all_reports['ConsensusFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
#selection_reports['ConsensusFilter'] = reports(mask, ~dict(clf.steps)['ConsensusFilter'.lower()].status, labels)
#
## MajorityVoteFilter
#MF = MajorityVoteFilter(filters)
#rfc = RandomForestClassifier(n_estimators=100)
#clf = make_pipeline(MF, rfc)
#clf.fit(X_train, y_train)
#all_reports['MajorityVoteFilter'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
#selection_reports['MajorityVoteFilter'] = reports(mask, ~dict(clf.steps)['MajorityVoteFilter'.lower()].status, labels)

####
# these methods are not working, the code does not correspond to the description
# of the algorithm in the paper. I could not understand its structure...
####

## YuanGuanZhu Majority
#YGZM = YuanGuanZhu(method='majority')
#rfc = RandomForestClassifier(n_estimators=100)
#clf = make_pipeline(YGZM, rfc)
#clf.fit(X, y, **{'yuanguanzhu__filters':filts})
#all_reports['YuanGuanZhu Majority'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
#selection_reports['YuanGuanZhu Majority'] = reports(mask, ~dict(clf.steps)['YuanGuanZhu'.lower()].status, labels)
#
## YuanGuanZhu Consensus
#YGZC = YuanGuanZhu(filts, method='consensus')
#rfc = RandomForestClassifier(n_estimators=100)
#clf = make_pipeline(YGZC, rfc)
#clf.fit(X_train, y_train)
#all_reports['YuanGuanZhu Consensus'] = reports(y_test, clf.predict(X_test), {0:0,1:1})
#selection_reports['YuanGuanZhu Consensus'] = reports(mask, ~dict(clf.steps)['YuanGuanZhu'.lower()].status, labels)
#
## composite filter
#compfilter = CompositeFilter()
#rfc = RandomForestClassifier(n_estimators=100)
#clf = make_pipeline(compfilter, rfc)
#clf.fit(X, y, **{'compositefilter__filters':filts})


#scores = {}
#for name, reps in all_reports.items():
#    scores[name] = reps[-1]
#
#pd.concat(scores).reset_index().pivot('level_0','level_1','Score')
