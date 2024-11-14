import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def AUC_CI(auc, label, alpha = 0.05):
    label = np.array(label) 
    n1, n2 = np.sum(label == 1), np.sum(label == 0)
    q1 = auc / (2-auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2))
    confidence_level = 1 - alpha
    z_lower, z_upper = norm.interval(confidence_level)
    lowerb, upperb = auc + z_lower * se, auc + z_upper * se
    return (lowerb, upperb)

def plot_AUC(ax, FPR, TPR, AUC, CI, label):
    label = '{}: {} ({}-{})'.format(str(label), round(AUC, 3), round(CI[0], 3), round(CI[1], 3))
    ax.plot(FPR, TPR, label = label)
    return ax