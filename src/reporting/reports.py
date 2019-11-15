import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def make_confusion_matrix(y_true, y_pred, target_names):
    """Generates a confusion matrix. Returns 1) Pandas Dataframe with confusion
    matrix, Producer's accuracy and User's accuracy and 2) Pandas Dataframe with
    Overall accuracy, F-Score, and G-mean score."""
    labels = list(target_names.keys())
    names = list(target_names.values())
    #labels.sort()
    cm = confusion_matrix(y_true, y_pred, labels=labels).T
    total_h = np.sum(cm, 0)
    total_v = np.sum(cm, 1)
    total   = sum(total_h)
    tp      = cm.diagonal()
    fp      = total_v-tp
    fn      = total_h-tp
    tn      = total-(fn+tp+fp)
    spec    = tn/(fp+tn)
    ua      = tp/(total_v+1e-100)
    pa      = tp/(total_h+1e-100)
    oa      = sum(tp)/total
    fscore  = 2*((np.mean(ua)*np.mean(pa))/(np.mean(ua)+np.mean(pa)))
    gmean   = np.sqrt(np.mean(pa)*np.mean(spec))
    core_cm = pd.DataFrame(index=names, columns=names, data=cm)\
                .append([
                    pd.Series(data=total_h, index=names, name='Total').astype(int),
                    pd.Series(data=pa, index=names, name='PA')
                    ])
    core_cm['Total'] = np.append(total_v, [total, np.nan])
    core_cm['Total'] = core_cm['Total'].map('{:,.0f}'.format).replace('nan', '')
    core_cm.loc['Total', names] = core_cm.loc['Total', names].astype(int).map('{:,.0f}'.format)
    core_cm['UA']    = np.append(ua, [np.nan,np.nan])
    core_cm['UA']    = core_cm['UA'].map('{:,.3f}'.format).replace('nan', '')
    core_cm.loc['PA', names] = core_cm.loc['PA', names].astype(float).map('{:,.3f}'.format)
    core_cm.loc[names, names] = core_cm.loc[names, names].applymap('{:,.0f}'.format)
    scores = pd.DataFrame(data= [oa, fscore, gmean], index=['ACCURACY', 'F-SCORE MACRO', 'G-MEAN MACRO'], columns=['Score'])
    return core_cm, scores


def reports(y_true, y_pred, target_names):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    labels = {k:target_names[k] for k in np.unique(y_true)}

    clf_report = pd.DataFrame(classification_report(y_true, y_pred, target_names=labels.values(), output_dict=True)).T
    conf_matrix, scores = make_confusion_matrix(y_true, y_pred, target_names=labels)
    return clf_report, conf_matrix, scores
