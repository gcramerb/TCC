import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def impute(data_missing, impute_type, idx_missing = None):

    dim = data.shape[2]
    if  impute_type == 'mean_all':

        for i in range(data.shape[0]):
            idx_notM = list(set(range(dim))- set(idx_missing[i]))
            defautMeanX = np.mean(data_missing[i,0,idx_notM,0])
            defautMeanY = np.mean(data_missing[i,0,idx_notM,1])
            defautMeanZ = np.mean(data_missing[i,0,idx_notM,2])
            data_missing[i,0,idx_missing,0:3] = [defautMeanX,defautMeanX,defautMeanZ]

    return data_missing

