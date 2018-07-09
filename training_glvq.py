import numpy as np
import itertools
from sklearn_lvq import GlvqModel
from sklearn_lvq.utils import plot2d
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import pandas as pd
from sklearn.externals import joblib
import time
import warnings
warnings.filterwarnings("ignore")

source = pd.read_excel("Full Data.xlsx", sheet_name="Data Cleanse")
features = ['THETA1_RIGHT_HAND', 'THETA2_RIGHT_HAND', 'THETA1_LEFT_HAND',
       'THETA2_LEFT_HAND', 'THETA1_RIGHT_WRIST', 'THETA2_RIGHT_WRIST',
       'THETA1_LEFT_WRIST', 'THETA2_LEFT_WRIST', 'THETA1_RIGHT_ELBOW',
       'THETA2_RIGHT_ELBOW', 'THETA1_LEFT_ELBOW', 'THETA2_LEFT_ELBOW',
       'THETA1_RIGHT_SHOULDER', 'THETA2_RIGHT_SHOULDER',
       'THETA1_LEFT_SHOULDER', 'THETA2_LEFT_SHOULDER']
ket = ['# FRAME', 'FRAME_KINECT', 'TIME', 'PERAGA', 'KALIMAT', 'PERCOBAAN','KATA','KATA_ASLI']

# Ganti di Source Peraga untuk membedakan lelaki, wanita dan campur
source = source[(source["KATA"] != "transisi") & (source["PERAGA"] == 2)]
#source = source[(source["KATA"] != "transisi")]


# In[ ]:


no_percobaan = {1,2,3,4,5}
training = list(itertools.combinations(no_percobaan,3))
testing = []
for x in training:
    temp = set(x)
    testing.append(list(no_percobaan - temp))
print training

# In[ ]:


counter = 1
i = 1
print(training[i])
prototype = 12
modus_akurasi = True
#nama_file = 'kata_modus_mixed.xlsx'
#for i in range(0,10):
data_train = source[(source["PERCOBAAN"]==training[i][0]) | (source["PERCOBAAN"]==training[i][1]) | (source["PERCOBAAN"]==training[i][2])]
data_test = source[(source["PERCOBAAN"]==testing[i][0]) | (source["PERCOBAAN"]==testing[i][1])]
X_train = data_train[features].values
X_test = data_test[features].values
kata = data_train["KATA_ASLI"]
label = data_test["KATA_ASLI"].values
labels, levels = pd.factorize(kata)
y = labels
#for prototype in range(1,51):
start_time = time.clock()
glvq = GlvqModel(prototypes_per_class = prototype, random_state = 99)
duration = time.clock() - start_time
glvq.fit(X_train, y)
joblib.dump(glvq, 'female_mixed.pkl')
print levels

