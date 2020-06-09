import numpy as np

from numpy.random import seed
from numpy.random import rand
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer


def impute(data_missing, impute_type, idx_missing = None):

    # ***
    #Idxmissing Ã© uma lista de dimensÃ£o N (amostras) em que cada elemento Ã© uma lista de indices em que o velor Ã© nan no
    #dado em questÃ£o



    #****

    dim = data_missing.shape[2]
    if  impute_type == 'mean':
        for i in range(data_missing.shape[0]):
            idx_notM = list(set(range(dim)) - set(idx_missing[i]))
            defautMeanX = np.mean(data_missing[i, 0, idx_notM, 0])
            defautMeanY = np.mean(data_missing[i, 0, idx_notM, 1])
            defautMeanZ = np.mean(data_missing[i, 0, idx_notM, 2])
            data_missing[i,0,idx_missing[i],0:3] = [defautMeanX,defautMeanY,defautMeanZ]

    if  impute_type == 'mode':
        for i in range(data_missing.shape[0]):
            idx_notM = list(set(range(dim))- set(idx_missing[i]))
            defautModeX = statistics.mode(data_missing[i,0,idx_notM,0])
            defautModeY = statistics.mode(data_missing[i,0,idx_notM,1])
            defautModeZ = statistics.mode(data_missing[i,0,idx_notM,2])
            data_missing[i,0,idx_missing[i],0:3] = [defautModeX,defautModeY,defautModeZ]


    if impute_type == 'median':
        for i in range(data_missing.shape[0]):
            idx_notM = list(set(range(dim))- set(idx_missing[i]))
            defautMedianX = np.median(data_missing[i,0,idx_notM,0])
            defautMedianY = np.median(data_missing[i,0,idx_notM,1])
            defautMedianZ = np.median(data_missing[i,0,idx_notM,2])
            data_missing[i,0,idx_missing[i],0:3] = [defautMedianX,defautMedianY,defautMedianZ]

    if impute_type == 'last_value':
        for i in range(data_missing.shape[0]):
            print(idx_missing[i][0] - 1)
            idx_notM = list(set(range(dim)) - set(idx_missing[i]))
            lastVx = data_missing[i,0,idx_missing[i][0]-1,0]
            lastVy = data_missing[i,0,idx_missing[i][0]-1,1]
            lastVz = data_missing[i,0,idx_missing[i][0]-1,2]
            data_missing[i, 0, idx_missing[i], 0:3] = [lastVx, lastVy, lastVz]
    if impute_type == 'aleatory':
        seed(1)
        for i in range(data_missing.shape[0]):
            n = len(idx_missing[i])
            minX = np.nanmin(data_missing[i,0,:,0])
            minY = np.nanmin(data_missing[i,0,:,1])
            minZ = np.nanmin(data_missing[i,0,:,2])
            maxX = np.nanmax(data_missing[i,0,:,0])
            maxY= np.nanmax(data_missing[i,0,:,1])
            maxZ =np.nanmax(data_missing[i,0,:,2])
            x = minX + (rand(n) * (maxX - minX))
            y = minY + (rand(n) * (maxY - minY))
            z = minZ + (rand(n) * (maxZ - minZ))
            data_missing[i, 0, idx_missing[i], 0] = x
            data_missing[i, 0, idx_missing[i], 1] = y
            data_missing[i, 0, idx_missing[i], 2] = z



    if impute_type == 'interpolation':
        for i in range(data_missing.shape[0]):
            data_missing[i,0,:,0 ] = pd.Series(data_missing[i,0,:,0 ]).interpolate()
            data_missing[i, 0, :, 1] = pd.Series(data_missing[i, 0, :, 1]).interpolate()
            data_missing[i, 0, :,2] = pd.Series(data_missing[i, 0, :, 2]).interpolate()
    if impute_type == 'knn':
        for i in range(data_missing.shape[-1]):
            imputer = KNNImputer(n_neighbors=5, weights="uniform")
            data_missing[:,0,:,i] = imputer.fit_transform(data_missing[:,0,:,i])


    return data_missing

