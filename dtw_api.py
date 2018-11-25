import pandas as pd
import numpy as np
from fastdtw import fastdtw
from sklearn.externals import joblib
from scipy.spatial.distance import euclidean


# LOAD DATA TRAIN

data_train_pria = joblib.load('pria_dtw.joblib')
data_train_wanita = joblib.load('wanita_dtw.joblib')
data_train_campur = joblib.load('campur_dtw.joblib')


# AMBIL DATA TRAIN

y_pria = data_train_pria["FITUR"].values
y_wanita = data_train_wanita["FITUR"].values
y_campur = data_train_campur["FITUR"].values


# PROSES DTW

temp = []
for k in range(0,len(y_pria)):
    distance, path = fastdtw(x, y_pria[k], dist=euclidean)
    temp.append(distance)
hasil_pria = np.argmin(temp)
label = data_train_pria["KATA"].reset_index()
pred_pria = label["KATA"][hasil_pria]

temp = []
for k in range(0,len(y_wanita)):
    distance, path = fastdtw(x, y_wanita[k], dist=euclidean)
    temp.append(distance)
hasil_wanita = np.argmin(temp)
label = data_train_wanita["KATA"].reset_index()
pred_wanita = label["KATA"][hasil_wanita]

temp = []
for k in range(0,len(y_campur)):
    distance, path = fastdtw(x, y_campur[k], dist=euclidean)
    temp.append(distance)
hasil_campur = np.argmin(temp)
label = data_train_campur["KATA"].reset_index()
pred_campur = label["KATA"][hasil_campur]

