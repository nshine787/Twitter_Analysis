# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:06:59 2019

@author: Nathan

Dataset can be found on Kaggle: https://www.kaggle.com/crowdflower/twitter-user-gender-classification
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from timeit import default_timer as timer

df = pd.read_csv("gender_data2.csv", encoding="latin1", index_col=0)
df.drop(['name', 'sidebar_color'], axis=1, inplace=True)

stoplist_name = "stoplist.txt"
with open(stoplist_name) as stop_file:
    stoplist = stop_file.read().splitlines()
stoplist = [word.replace("'", "") for word in stoplist]

'''
Removing the URLs, usernames, and emojis from the tweets
'''
tweets = df['text'].tolist()
tweets = [re.sub(r"http\S+|@\S+|[^\x00-\x7F]+|amp", "",tweet).replace("'", "") for tweet in tweets]
df['text'] = tweets

'''
Relabeling the gender in terms of an integer
'''
df.loc[df['gender'] == 'brand', 'gender',] = 0
df.loc[df['gender'] == 'female', 'gender',] = 1
df.loc[df['gender'] == 'male', 'gender',] = 2
df = df[df['gender'] != 'unknown']
df.dropna(inplace=True)

dfx = df['text']
dfy = df['gender']
cv = CountVectorizer(stop_words=stoplist)
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=4)
x_traincv = cv.fit_transform(x_train)
train_arr = x_traincv.toarray()

x_testcv=cv.transform(x_test)

mnb = MultinomialNB()
y_train = y_train.astype('int')
mnb.fit(x_traincv,y_train)

testmessage=x_test.iloc[0]
predictions=mnb.predict(x_testcv)

ystar = np.array(y_test)
acc_array = predictions==ystar
accuracy = np.sum(acc_array) / len(acc_array)

weights = mnb.coef_
bweights = weights[0].argsort()
fweights = weights[1].argsort()
mweights = weights[2].argsort()

counts = mnb.feature_count_
bcounts = counts[0]
fcounts = counts[1]
mcounts = counts[2]

fnames = cv.get_feature_names()

'''
Finding highest strength words
tff - top female factors
tmf - top male factors
tbf - top brand factors
'''
min_count = 3
tff, tmf, tbf = [], [], []
for i in reversed(fweights):
    if fcounts[i] > min_count and fnames[i][0] != '_':
        tff.append(fnames[i])
    if len(tff) > 11:
        break
for i in reversed(mweights):
    if mcounts[i] > min_count and fnames[i][0] != '_':
        tmf.append(fnames[i])
    if len(tmf) > 11:
        break
for i in reversed(bweights):
    if bcounts[i] > min_count and fnames[i][0] != '_':
        tbf.append(fnames[i])
    if len(tbf) > 11:
        break

#print(tff)
#print(tmf)
#print(tbf)
#start = timer()
#kmeans = KMeans(n_clusters=3, max_iter=50)
#kmeans.fit(train_arr)
#centroid = kmeans.cluster_centers_
#labels = kmeans.labels_
#ytstar = np.array(y_train)
#kacc_arr = labels==ytstar
#kacc = np.sum(kacc_arr)/len(kacc_arr)
#end = timer()
#
#print("TIME: ", end-start)