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

# read data in as data frame
st = pd.read_csv('/Users/XS/Desktop/Project/HRC_train.tsv', sep = 'str("") | str("\t")',  names=['content'], engine='python')
df = pd.DataFrame(st.content.str.split('"\t"',1).tolist(), columns = ['label','content'])

# label
for i in df.index:
    df.label[i] = df.label[i][1]
label = np.array(df.label)


# get rid of punctuations and numbers
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
translator_2 = {ord(k): None for k in digits}
df['upd_content'] = df.content
df['content_non_punc'] = df.content

for i in df.index:
	df['upd_content'][i] = df.content[i].translate(translator)
	df['content_non_punc'][i] = df['upd_content'][i].translate(translator_2)

# tokenize all content
df['token'] = df['content_non_punc'].apply(nltk.word_tokenize)

# steming correction
df['new_token2'] = df.token
stop_words = set(stopwords.words('english'))
stop_words.update(['fw', 'subject', 'case', 'doc', 'unclassified', 'a','b','c','d','e','f','g','h''i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
porter = PorterStemmer()
for i in df.index:
	email = df.token[i] 
	list_of_words = [porter.stem(j.lower()) for j in email if (j.lower() not in stop_words)]
	df.new_token2[i] = list_of_words

# create a dictionary in the form {word1:count1,...}
# dictionary = df.new_token.apply(Counter)
dictionary2 = df.new_token2.apply(Counter)
# print(dictionary)

# convert the dictionary into a data frame
v = DictVectorizer(sparse = False)
matrix_temp = v.fit_transform(dictionary2)
# print("only stemming not enchant")
print(matrix_temp.shape)
names_temp = v.get_feature_names()
data_temp = pd.DataFrame(matrix_temp,columns=names_temp)
colsum_temp = data_temp.sum(axis=0)

num = 0
# print("words not recognized by enchant after only stemming")
candidates =[]
for i in range(len(colsum_temp)):
	# if colsum_temp[i]>10 and d.check(names_temp[i])==False and (names_temp[i] not in somelist):
	if colsum_temp[i]> 10:

		# print(names_temp[i],colsum_temp[i])
		candidates.append(names_temp[i])
		num +=1
print(num)
# data_candidates = data_temp.loc[:,candidates]
# data_candidates['biaoqian'] = label
# data_candidates.to_csv('/Users/XS/Desktop/data_candidates.csv', index=False, header=True, sep=',')


#************************************************************
training = data_temp.loc[:,candidates]
# training = pd.concat([punctuation,training],axis = 1)
training_matrix = np.array(training)
print("training matrix shape")
print(training_matrix.shape)

RF = RandomForestClassifier(n_estimators=500,criterion='entropy',max_features=200,max_depth=100,oob_score=True)
RF.fit(training_matrix,label)
# print(RF.feature_importances_)
print(RF.oob_score_)


#*********************************************************************
# read data in as data frame
df = pd.read_csv('/Users/XS/Desktop/Project/HRC_test.tsv', sep = 'str("\t")',  names=['content'], engine='python')

# label
# for i in df.index:
#     df.label[i] = df.label[i][1]
# label = np.array(df.label)


# get rid of punctuations and numbers
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
translator_2 = {ord(k): None for k in digits}
df['upd_content'] = df.content
df['content_non_punc'] = df.content

for i in df.index:
	df['upd_content'][i] = df.content[i].translate(translator)
	df['content_non_punc'][i] = df['upd_content'][i].translate(translator_2)

# tokenize all content
df['token'] = df['content_non_punc'].apply(nltk.word_tokenize)

# steming correction
df['new_token'] = df.token
stop_words = set(stopwords.words('english'))
stop_words.update(['fw', 'subject', 'case', 'doc', 'unclassified', 'a','b','c','d','e','f','g','h''i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
porter = PorterStemmer()
for i in df.index:
	email = df.token[i] 
	list_of_words = [porter.stem(j.lower()) for j in email if (j.lower() not in stop_words)]
	df.new_token[i] = list_of_words

# create a dictionary in the form {word1:count1,...}
# dictionary = df.new_token.apply(Counter)
dictionary = df.new_token.apply(Counter)
# print(dictionary)

# convert the dictionary into a data frame
ve = DictVectorizer(sparse = False)
freq = ve.fit_transform(dictionary)
# print("only stemming not enchant")
print(freq.shape)
test_names = ve.get_feature_names()
test_df = pd.DataFrame(freq,columns=test_names)
test_df['nothisword']=np.zeros((389))

trainlist = training.dtypes.index
testlist =[]
for name in trainlist:
	if name in test_names:
		testlist.append(name)
	else:
		testlist.append("nothisword")
test = test_df.loc[:,testlist]
test_matrix = np.array(test)
print("test matrix shape")
print(test_matrix.shape)

pred = RF.predict(test_matrix)
print(pred.shape)

file = open("pred.txt", "w")
for i in range(389):
	file.write("{0}\n".format(pred[i]))
file.close()




























