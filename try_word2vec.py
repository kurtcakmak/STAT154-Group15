# from pyspark.ml.feature import HashingTF, IDF, Word2Vec
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
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# read data in as data frame
st = pd.read_csv('/Users/XS/Desktop/Project/HRC_train.tsv', sep = 'str("") | str("\t")',  names=['content'], engine='python')
df = pd.DataFrame(st.content.str.split('"\t"',1).tolist(), columns = ['label','content'])

# label
for i in df.index:
    df.label[i] = df.label[i][1]
label = np.array(df.label)

'''
for j in [x for x in range(3505) if x not in [24,44,81,100,119,132,137,175,193,195,214,215,225,236,237,241]]:
    df.content[j] = re.search(r'subject: (.*?) (u.s. department of state case no.)', df.content[j]).group(1)
'''


# get rid of punctuations and numbers
translator = str.maketrans({key: None for key in string.punctuation})
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
	list_of_words = [porter.stem(j.lower()) for j in email if j.lower() not in stop_words]
	df.new_token[i] = list_of_words
sentences = df['new_token']


df = pd.read_csv('/Users/XS/Desktop/Project/HRC_test.tsv', sep = 'str("\t")',  names=['content'], engine='python')


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
testsentences = df['new_token']



# word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="new_token", outputCol="result")
# model = word2Vec.fit(df)
# result = model.transform(df)
# for feature in result.select("result").take(3):
#     print(feature)
num_features = 40    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 15          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
# print(sentences[0])

model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
# print(model["war"])


matrix =[]
for i in range(3505):
	k=0
	array = np.zeros((num_features))
	# print(sentences[i])
	for word in sentences[i]:
		if word in model.vocab:
			array = array + model[word]
			k=k+1
	# print(k)
	array = array/k
	# print(array)
	matrix.append(array)
	# print(i)
matrix=np.matrix(matrix)
print(matrix.shape)

testmatrix =[]
for i in range(389):
	k=0
	array = np.zeros((num_features))
	# print(sentences[i])
	for word in testsentences[i]:
		if word in model.vocab:
			array = array + model[word]
			k=k+1
	# print(k)
	array = array/k
	# print(array)
	testmatrix.append(array)
	# print(i)
testmatrix=np.matrix(testmatrix)
print(testmatrix.shape)

train = pd.read_csv("/Users/XS/Desktop/train.csv")
test = pd.read_csv("/Users/XS/Desktop/test.csv")

m1 = np.array(train)
m2 = np.array(test)

print(m1.shape)
print(m2.shape)

X_train = np.append(m1[:,:6497],matrix,axis=1)
X_test = np.append(m2[:,:6497],testmatrix,axis=1)

RF = RandomForestClassifier(n_estimators=500,criterion='entropy',max_features=100,max_depth=100,oob_score=True)
# RF.fit(m1[:,:6497],label)
RF.fit(X_train,label)
# print(RF.feature_importances_)
print(RF.oob_score_)
