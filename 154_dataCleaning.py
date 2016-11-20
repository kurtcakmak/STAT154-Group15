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

#keys = list(dictionary[0].keys())
#print(len(keys))
#dic = dictionary[0]
#keys1 = keys[0]
#print(dic[keys1])

#print(dictionary[0])
#df.content[0] = df.new_token[0]
#print(df)
#print(dictionary)

# sum up all counters
sum_dictionary = dictionary[0]
for index in range(1, len(df.index)):
    sum_dictionary = sum_dictionary + dictionary[index]
unique_words = sum_dictionary.keys()
#print(sum_dictionary)
#print(len(df.index))
#print(len(sum_dictionary))
#print(len(unique_words))
#print(unique_words)

# convert the dictionary into a data frame
#w = len(df.index) + 1
#h = len(unique_words)
#w=3
#h=5
#Matrix = [[0 for x in range(w)] for y in range(h)]
unique_words = list(unique_words)
Matrix = np.zeros((len(df.index),len(unique_words)))
#print(Matrix)
#unique_words = unique_words[0:5]
#print(Matrix[0])
#print(len(Matrix[0]))
#print(len(Matrix[1:w]))
#np.set_printoptions(threshold=np.nan)
def makeFrame(Matrix, dictionary, unique_words):
	for j in range(len(df.index)):
		for key in list(dictionary[j].keys()):
			for i in range(len(unique_words)):
				if str(unique_words[i]) == str(key):
					count = dictionary[j][key]
					Matrix[j,i] = int(count)					
	return(Matrix)

def makeFrame2(Matrix, dictionary, unique_words):
	for key in list(dictionary[0].keys()):
		for i in range(len(unique_words)):
			if str(unique_words[i]) == str(key):
					#print(dictionary[0][key])
				count = dictionary[0][key]
				Matrix[0,i] = int(count)
					#print(Matrix[0,i])
	return(Matrix)

#a=makeFrame(Matrix, dictionary, unique_words)
a=makeFrame(Matrix, dictionary, unique_words)
with open("output1.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(unique_words)
with open("output2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(a)
#data_frame = pd.DataFrame(a)
#data_frame.to_csv("154data.csv", sep='\t')

#print(a)
#print(len(a[0]))
b=a[0]
print(b[b>0])
#print(len(a))

count = 0
for i in range(len(a[0])):
	if a[0,i] > 0:
		count += 1
print(count)
print(len(dictionary[0].keys()))


# analyze data frame to find 1power feature
