from sklearn_lvq import GlvqModel
from sklearn.externals import joblib
from fastdtw import fastdtw
from hmmlearn import hmm
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
import json
import editdistance
import warnings
warnings.filterwarnings("ignore")

levels = ['hari', 'ini', 'apa', 'pohon', 'kelapa', 'itu', 'tinggi', 'atau',
       'tidak', 'pendek', 'tetapi', 'ibu', 'ku', 'kuat', 'gemuk', 'aku',
       'memukul', 'dia', 'dipukul', 'kamu', 'harus', 'makan', 'keluar', 'dari',
       'sini']

def predict_glvq(X_test):	   
    tes = joblib.load('models/glvq/mixed.pkl')
    dir(tes)
    label_predict = [levels[i] for i in tes.predict(X_test)]
    score = np.amin(tes._compute_distance(X_test),axis=1)
    
    tes = joblib.load('models/glvq/male_mixed.pkl')
    label_predict_m = [levels[i] for i in tes.predict(X_test)]
    score_m = np.amin(tes._compute_distance(X_test),axis=1)
    
    tes = joblib.load('models/glvq/female_mixed.pkl')
    label_predict_f = [levels[i] for i in tes.predict(X_test)]
    score_f = np.amin(tes._compute_distance(X_test),axis=1)
    
    return [(label_predict[0], score[0]), (label_predict_m[0], score_m[0]), (label_predict_f[0], score_f[0])]

def predict_dtw(X_test):
    # LOAD DATA TRAIN
    data_train_pria = joblib.load('models/dtw/pria_dtw.joblib')
    data_train_wanita = joblib.load('models/dtw/wanita_dtw.joblib')
    data_train_campur = joblib.load('models/dtw/campur_dtw.joblib')

    # AMBIL DATA TRAIN
    y_pria = data_train_pria["FITUR"].values
    y_wanita = data_train_wanita["FITUR"].values
    y_campur = data_train_campur["FITUR"].values

    # PROSES DTW
    temp = []
    for k in range(0,len(y_pria)):
        distance, path = fastdtw(X_test, y_pria[k], dist=euclidean)
        temp.append(distance)
    hasil_pria = np.argmin(temp)
    label = data_train_pria["KATA"].reset_index()
    pred_pria = label["KATA"][hasil_pria]

    temp = []
    for k in range(0,len(y_wanita)):
        distance, path = fastdtw(X_test, y_wanita[k], dist=euclidean)
        temp.append(distance)
    hasil_wanita = np.argmin(temp)
    label = data_train_wanita["KATA"].reset_index()
    pred_wanita = label["KATA"][hasil_wanita]

    temp = []
    for k in range(0,len(y_campur)):
        distance, path = fastdtw(X_test, y_campur[k], dist=euclidean)
        temp.append(distance)
    hasil_campur = np.argmin(temp)
    label = data_train_campur["KATA"].reset_index()
    pred_campur = label["KATA"][hasil_campur]
    
    return [pred_campur, pred_pria, pred_wanita]

def predict_hmm(X_test):
    # LOAD DATA TRAIN
    data_train_pria = joblib.load("models/hmm/data_train_laki.pkl")
    model_pria = joblib.load("models/hmm/model_hmm_laki.pkl")
    data_train_wanita = joblib.load("models/hmm/data_train_wanita.pkl")
    model_wanita = joblib.load("models/hmm/model_hmm_wanita.pkl")
    data_train_mixed = joblib.load("models/hmm/data_train_mixed.pkl")
    model_mixed = joblib.load("models/hmm/model_hmm_mixed.pkl")

    hasil_hmm_pria = ''.join(str(x) for x in model_pria.predict(X_test))
    temp = []
    for item in data_train_pria["HIDDEN_STATE_PREDICT"]:
        temp.append(editdistance.eval(hasil_hmm_pria, item)/max(len(hasil_hmm_pria),len(item)))
    index_pred_pria = np.argmin(temp)
    jarak_pria = temp[index_pred_pria]
    label_pred_pria = data_train_pria["KATA"].iloc[index_pred_pria]
    
    hasil_hmm_wanita = ''.join(str(x) for x in model_wanita.predict(X_test))
    temp = []
    for item in data_train_wanita["HIDDEN_STATE_PREDICT"]:
        temp.append(editdistance.eval(hasil_hmm_wanita, item)/max(len(hasil_hmm_wanita),len(item)))
    index_pred_wanita = np.argmin(temp)
    jarak_wanita = temp[index_pred_wanita]
    label_pred_wanita = data_train_wanita["KATA"].iloc[index_pred_wanita]
    
    hasil_hmm_mixed = ''.join(str(x) for x in model_mixed.predict(X_test))
    temp = []
    for item in data_train_mixed["HIDDEN_STATE_PREDICT"]:
        temp.append(editdistance.eval(hasil_hmm_mixed, item)/max(len(hasil_hmm_mixed),len(item)))
    index_pred_mixed = np.argmin(temp)
    jarak_mixed = temp[index_pred_mixed]
    label_pred_mixed = data_train_mixed["KATA"].iloc[index_pred_mixed]
    
    return [label_pred_mixed, label_pred_pria, label_pred_wanita]

    
if __name__ == '__main__':
    print ('Debugging...')
    with open('data/data-20182312160847-hari.json') as f:
        data = json.load(f)
        coll = []
        coll_m = []
        coll_f = []
        lbl = []
        lbl_m = []
        lbl_f = []
        arr = []
        arr_m = []
        arr_f = []
        for X_test in data:
            dt = predict([[X_test['RIGHT_HAND'][0], X_test['RIGHT_HAND'][1], X_test['LEFT_HAND'][0], X_test['LEFT_HAND'][1], 
            X_test['RIGHT_WRIST'][0], X_test['RIGHT_WRIST'][1], X_test['LEFT_WRIST'][0], X_test['LEFT_WRIST'][1],
            X_test['RIGHT_ELBOW'][0], X_test['RIGHT_ELBOW'][1], X_test['LEFT_ELBOW'][0], X_test['LEFT_ELBOW'][1],
            X_test['RIGHT_SHOULDER'][0], X_test['RIGHT_SHOULDER'][1], X_test['LEFT_SHOULDER'][0], X_test['LEFT_SHOULDER'][1]]])
            # print dt
            coll.append(float(dt[0][1]))
            coll_m.append(float(dt[1][1]))
            coll_f.append(float(dt[2][1]))
            lbl.append(dt[0][0])
            lbl_m.append(dt[1][0])
            lbl_f.append(dt[2][0])
            arr.append((dt[0][0], float(dt[0][1])))
            arr_m.append((dt[1][0], float(dt[1][1])))
            arr_f.append((dt[2][0], float(dt[2][1])))
        np_arr = np.array(arr, dtype=[('label', 'S25'), ('dist', float)])
        s_arr = np.sort(np_arr, order='dist')
        np_arr_m = np.array(arr_m, dtype=[('label', 'S25'), ('dist', float)])
        s_arr_m = np.sort(np_arr_m, order='dist')
        np_arr_f = np.array(arr_f, dtype=[('label', 'S25'), ('dist', float)])
        s_arr_f = np.sort(np_arr_f, order='dist')
        print s_arr[0:14]
        print s_arr_m[0:14]
        print s_arr_f[0:14]
        print 'Predict result mixed : {}\nPredict result male  : {}\nPredict result female: {}'.format(
        lbl[np.argmin(coll)], lbl_m[np.argmin(coll)], lbl_f[np.argmin(coll)]
        )