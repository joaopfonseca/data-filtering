import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.models.AutoEncoder import AEMLPClassifier
from src.experiment.utils import make_multiclass_noise
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import keras.backend as K

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


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
results = []
for train, test in cv.split(X, y):
    clf = AEMLPClassifier(batch_size=32, epochs=300)
    X_noisy, y_noisy = make_multiclass_noise(transfer_map, noise_level=0.05).fit_resample(X[train], y[train])
    clf.fit(X_noisy, y_noisy)
    y_pred = clf.predict(X[test])
    results.append(accuracy_score(y[test], y_pred))
    K.clear_session()
