from datetime import datetime
from scipy.stats import mode
import socket
import json
import predict_mixed
import warnings
warnings.filterwarnings("ignore")

HOST = ''
PORT = 9876
ADDR = (HOST,PORT)
BUFSIZE = 4096

serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

serv.bind(ADDR)
serv.listen(5)

if __name__ == '__main__':
	print ('listening ...')	
	
	try:
		while True:
			conn, addr = serv.accept()
			print ('client connected ... ', addr)
			pData = []
			
			while True:
				data = conn.recv(BUFSIZE)
				if not data: break
				pData.append(json.loads(data))
			
			fileID = "{:%Y%M%d%H%m%S}".format(datetime.now())
			with open("data/data-{}.json".format(fileID), 'w') as outfile:  
				json.dump(pData, outfile)
			
			coll = []
			coll_m = []
			coll_f = []
			for X_test in pData:
				dt = predict_mixed.predict([[X_test['RIGHT_HAND'][0], X_test['RIGHT_HAND'][1], X_test['LEFT_HAND'][0], X_test['LEFT_HAND'][1], 
				X_test['RIGHT_WRIST'][0], X_test['RIGHT_WRIST'][1], X_test['LEFT_WRIST'][0], X_test['LEFT_WRIST'][1],
				X_test['RIGHT_ELBOW'][0], X_test['RIGHT_ELBOW'][1], X_test['LEFT_ELBOW'][0], X_test['LEFT_ELBOW'][1],
				X_test['RIGHT_SHOULDER'][0], X_test['RIGHT_SHOULDER'][1], X_test['LEFT_SHOULDER'][0], X_test['LEFT_SHOULDER'][1]]])
				coll.append(dt[0])
				coll_m.append(dt[1])
				coll_f.append(dt[2])
				print dt
				
			# print coll
			print ('Predict result mixed : ', mode(coll))
			print ('Predict result male  : ', mode(coll_m))
			print ('Predict result female: ', mode(coll_f))
			dir(mode(coll))
			
			with open("result/result-{}.txt".format(fileID), 'w') as resfile:  
				resfile.write('Predict result mixed : {} with frequency {}/{}\n'.format(mode(coll).mode[0][0], mode(coll).count[0][0], len(coll)))
				resfile.write('Predict result male  : {} with frequency {}/{}\n'.format(mode(coll_m).mode[0][0], mode(coll_m).count[0][0], len(coll_m)))
				resfile.write('Predict result female: {} with frequency {}/{}\n'.format(mode(coll_f).mode[0][0], mode(coll_f).count[0][0], len(coll_f)))
				print ('Saved to result/result-{}.txt'.format(fileID))

			conn.close()
			print ('client disconnected')
	except KeyboardInterrupt:
		pass
	
	serv.shutdown(0)
	serv.close()