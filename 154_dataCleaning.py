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
porter = PorterStemmer()
for i in df.index:
	email = df.token[i] 
	list_of_words = [porter.stem(j.lower()) for j in email if j.lower() not in stop_words]
	df.new_token[i] = list_of_words
print(df.new_token)

# create a dictionary in the form {word1:count1,...}
dictionary = df.new_token.apply(Counter)
print(dictionary[0])
#df.content[0] = df.new_token[0]
#print(df)
#print(dictionary)

# convert the dictionary into a data frame
#dictionary.df = pd.DataFrame(dictionary, columns = ['label',)

# analyze data frame to find power feature
