import numpy as np
import json
from sklearn_lvq import GlvqModel
import pandas as pd
from sklearn.externals import joblib

levels = ['hari', 'ini', 'apa', 'pohon', 'kelapa', 'itu', 'tinggi', 'atau',
       'tidak', 'pendek', 'tetapi', 'ibu', 'ku', 'kuat', 'gemuk', 'aku',
       'memukul', 'dia', 'dipukul', 'kamu', 'harus', 'makan', 'keluar', 'dari',
       'sini']

def predict(X_test):	   
	tes = joblib.load('models/mixed.pkl')
	dir(tes)
	label_predict = [levels[i] for i in tes.predict(X_test)]
	score = np.amin(tes._compute_distance(X_test),axis=1)
	
	tes = joblib.load('models/male_mixed.pkl')
	label_predict_m = [levels[i] for i in tes.predict(X_test)]
	score_m = np.amin(tes._compute_distance(X_test),axis=1)
	
	tes = joblib.load('models/female_mixed.pkl')
	label_predict_f = [levels[i] for i in tes.predict(X_test)]
	score_f = np.amin(tes._compute_distance(X_test),axis=1)
	
	return [(label_predict, score), (label_predict_m, score_m), (label_predict_f, score_f)]
	
if __name__ == '__main__':
	print ('Debugging...')
	with open('data/data-20180014150703.json') as f:
		data = json.load(f)
		coll = []
		coll_m = []
		coll_f = []
		for X_test in data:
			dt = predict([[X_test['RIGHT_HAND'][0], X_test['RIGHT_HAND'][1], X_test['LEFT_HAND'][0], X_test['LEFT_HAND'][1], 
			X_test['RIGHT_WRIST'][0], X_test['RIGHT_WRIST'][1], X_test['LEFT_WRIST'][0], X_test['LEFT_WRIST'][1],
			X_test['RIGHT_ELBOW'][0], X_test['RIGHT_ELBOW'][1], X_test['LEFT_ELBOW'][0], X_test['LEFT_ELBOW'][1],
			X_test['RIGHT_SHOULDER'][0], X_test['RIGHT_SHOULDER'][1], X_test['LEFT_SHOULDER'][0], X_test['LEFT_SHOULDER'][1]]])
			print dt
		# print (predict(X_test))