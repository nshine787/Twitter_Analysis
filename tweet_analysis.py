# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:06:59 2019

@author: Nathan

Dataset can be found on Kaggle: https://www.kaggle.com/crowdflower/twitter-user-gender-classification
"""

import functions
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from timeit import default_timer as timer

kaggleDatasetName = 'crowdflower/twitter-user-gender-classification'
# functions.downloadDataset(kaggleDatasetName)
datasetFileName = functions.getDatasetFileName(kaggleDatasetName)    
df = pd.read_csv(datasetFileName, encoding="latin1", index_col=0)
interestingColumns = ['gender', 'description','text','sidebar_color','tweet_count']
df = df[interestingColumns]

stoplist_name = "stoplist.txt"
with open(stoplist_name) as stop_file:
    stoplist = stop_file.read().splitlines()
stoplist = [word.replace("'", "") for word in stoplist]

'''
Removing the URLs, usernames, and emojis from the tweets
'''
df = df.replace(np.nan, '', regex=True)

tweets = df['text'].tolist()
descriptions = df['description'].tolist()

tweets = functions.cleanDataset(tweets)
descriptions = functions.cleanDataset(descriptions)

df['text'] = tweets
df['description'] = descriptions
descriptions[:5]

# tweets = [re.sub(r"http\S+|@\S+|[^\x00-\x7F]+|amp", "",tweet).replace("'", "") for tweet in tweets]


'''
Relabeling the gender in terms of an integer
'''
df.loc[df['gender'] == 'brand', 'gender',] = 0
df.loc[df['gender'] == 'female', 'gender',] = 1
df.loc[df['gender'] == 'male', 'gender',] = 2
df = df[~df['gender'].isin(['unknown',''])]

dfx = df['description']
dfy = df['gender']
cv = CountVectorizer(stop_words=stoplist)
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=4)
x_traincv = cv.fit_transform(x_train)
train_arr = x_traincv.toarray()

x_testcv=cv.transform(x_test)

mnb = MultinomialNB()
y_train = y_train.values.astype(np.int64)
mnb.fit(x_traincv,y_train)

predictions=mnb.predict(x_testcv)

ystar = np.array(y_test)
acc_array = predictions==ystar
accuracy = np.sum(acc_array) / len(acc_array)

fnames = cv.get_feature_names()
weights = mnb.coef_
counts = mnb.feature_count_

mnbResultsDict = functions.constructResultsDictionary(mnb, fnames)

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