import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from src.experiment.utils import make_multiclass_noise
from src.models.oversampling import DenoisedGeometricSMOTE
from sklearn.ensemble import RandomForestClassifier

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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
results = []
for train, test in cv.split(X, y):
    # introduce noise
    X_noisy, y_noisy = make_multiclass_noise(transfer_map, noise_level=0.05).fit_resample(X[train], y[train])

    # preprocess noise
    X_denoised, y_denoised = DenoisedGeometricSMOTE(random_state=random_state).fit_resample(X_noisy,y_noisy)

    # train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_denoised, y_denoised)
    y_pred = clf.predict(X[test])
    results.append(accuracy_score(y[test], y_pred))
