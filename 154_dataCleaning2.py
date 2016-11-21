# -*- coding: utf-8 -*-
"""
Statistics 154 Final Project 

Created on Mon Nov 14 18:13:48 2016

@author: yingyingli
"""

import nltk
import pandas as pd
import re
from collections import Counter
import string
from string import digits
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer

# read data in as data frame
st = pd.read_csv('/Users/kurtcakmak1/Desktop/Stat154/FinalProject/HRC_train.tsv', sep = 'str("") | str("\t")',  names=['content'], engine='python')
df = pd.DataFrame(st.content.str.split('"\t"',1).tolist(), columns = ['label','content'])
#print(df)

# label
for i in df.index:
    df.label[i] = df.label[i][1]
'''
for j in [x for x in range(3505) if x not in [24,44,81,100,119,132,137,175,193,195,214,215,225,236,237,241]]:
    df.content[j] = re.search(r'subject: (.*?) (u.s. department of state case no.)', df.content[j]).group(1)
'''
#print(df.label[0:10])
#print(df.content[0])

# get rid of punctuations and numbers
translator = str.maketrans({key: None for key in string.punctuation})
translator_2 = {ord(k): None for k in digits}
df['upd_content'] = df.content
df['content_non_punc'] = df.content

for i in df.index:
	df['upd_content'][i] = df.content[i].translate(translator)
	df['content_non_punc'][i] = df['upd_content'][i].translate(translator_2)


# tokenize all content
df['token'] = df.content_non_punc.apply(nltk.word_tokenize)

#remove stopwords + stemming
df['new_token'] = df.token
stop_words = set(stopwords.words('english'))
stop_words.update(['fw', 'subject', 'case', 'doc', 'unclassified', 'a','b','c','d','e','f','g','h''i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
porter = PorterStemmer()
for i in df.index:
	email = df.token[i] 
	list_of_words = [porter.stem(j.lower()) for j in email if j.lower() not in stop_words]
	df.new_token[i] = list_of_words
#print(df.new_token[0])
#print(len(df.new_token[0]))

#Manually remove single letters, insignificant words
#df['test'] = df.new_token
#stop_words.update(['fw', 'unclassifi', 'a','b','c','d','e','f','g','h''i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
#for i in df.index:
#	email = df.new_token[i] #
#	list_of_words = [j for j in email if j not in stop_words]
#	df.test[i] = list_of_words	
#print(df.new_token[0])
#print(df.test[0])
#print(len(df.new_token[0]))
#print(len(df.test[0]))

# create a dictionary in the form {word1:count1,...}
dictionary = df.new_token.apply(Counter)

ve = DictVectorizer(sparse=False)
matrix = ve.fit_transform(dictionary)

unique_words = ve.get_feature_names()
print(matrix.shape)
print(unique_words)
with open("Output.txt", "w") as text_file:
    text_file.write(str(unique_words))

#    writer.writerows(unique_words)
with open("output2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(matrix)
#data_frame = pd.DataFrame(a)
#data_frame.to_csv("154data.csv", sep='\t')

#print(a)
#print(len(a[0]))


# analyze data frame to find 1power feature
