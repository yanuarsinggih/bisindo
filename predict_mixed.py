import numpy as np
import json
from sklearn_lvq import GlvqModel
import pandas as pd
from sklearn.externals import joblib

levels = ['hari', 'ini', 'apa', 'pohon', 'kelapa', 'itu', 'tinggi', 'atau',
       'tidak', 'pendek', 'tetapi', 'ibu', 'ku', 'kuat', 'gemuk', 'aku',
       'memukul', 'dia', 'dipukul', 'kamu', 'harus', 'makan', 'keluar', 'dari',
       'sini']
print(dir(GlvqModel))

def predict(X_test):	   
	tes = joblib.load('mixed.pkl')
	label_predict = [levels[i] for i in tes.predict(X_test)]
	
	tes = joblib.load('male_mixed.pkl')
	label_predict_m = [levels[i] for i in tes.predict(X_test)]
	
	tes = joblib.load('female_mixed.pkl')
	label_predict_f = [levels[i] for i in tes.predict(X_test)]
	
	return [label_predict, label_predict_m, label_predict_f]
	
if __name__ == '__main__':
	print ('Debugging...')
	with open('data.json') as f:
		data = json.load(f)
		for X_test in data:
			print predict([[X_test['RIGHT_HAND'][0], X_test['RIGHT_HAND'][1], X_test['LEFT_HAND'][0], X_test['LEFT_HAND'][1], 
			X_test['RIGHT_WRIST'][0], X_test['RIGHT_WRIST'][1], X_test['LEFT_WRIST'][0], X_test['LEFT_WRIST'][1],
			X_test['RIGHT_ELBOW'][0], X_test['RIGHT_ELBOW'][1], X_test['LEFT_ELBOW'][0], X_test['LEFT_ELBOW'][1],
			X_test['RIGHT_SHOULDER'][0], X_test['RIGHT_SHOULDER'][1], X_test['LEFT_SHOULDER'][0], X_test['LEFT_SHOULDER'][1]]])
		# print (predict(X_test))