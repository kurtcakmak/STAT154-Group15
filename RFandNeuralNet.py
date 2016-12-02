from sklearn.feature_extraction import DictVectorizer
import nltk
import pandas as pd
import re
import string
from string import digits
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
import enchant
from sklearn.neural_network import MLPClassifier

train = pd.read_csv("/Users/XS/Desktop/train_final.csv")
test = pd.read_csv("/Users/XS/Desktop/test.csv")

colsum_temp = train.sum(axis=0)
headers = train.dtypes.index
num = 0
p = len(headers)

label = train['biaoqian']
label = np.array(label)

# print("words not recognized by enchant after only stemming")
candidates =[]
for i in range(p-1):
	# if colsum_temp[i]>10 and d.check(names_temp[i])==False and (names_temp[i] not in somelist):
	if colsum_temp[i]> 50:

		# print(names_temp[i],colsum_temp[i])
		candidates.append(headers[i])
		num +=1
print(num)

X_train=np.array(train.loc[:,candidates])
X_test=np.array(test.loc[:,candidates])


m1 = np.array(train)
m2 = np.array(test)

print(m1.shape)
print(m2.shape)

# X_train = m1[:,:6497]
# X_test = m2[:,:6497]

RF = RandomForestClassifier(n_estimators=500,criterion='entropy',max_features=250,max_depth=70,oob_score=True)
# RF.fit(m1[:,:6497],label)
RF.fit(X_train,train['biaoqian'])
# print(RF.feature_importances_)
print(RF.oob_score_)


num = X_train.shape[1]
XY = np.append(X_train,label.reshape(3505,1),axis = 1)
np.random.shuffle(XY)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(500,), random_state=1)
clf.fit(XY[:2804,:num], XY[:2804,num])  

# MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#        beta_1=0.9, beta_2=0.999, early_stopping=False,
#        epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#        warm_start=False)
print(clf.n_layers_)
print(clf.score(XY[2804:,:num], XY[2804:,num]))






