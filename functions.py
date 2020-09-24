# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:48:50 2020

@author: Nathan
"""
import kaggle
import re

'''
Downloads the dataset from Kaggle given a name
Returns the name of the (first) file
'''
def downloadDataset(datasetName):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(datasetName, unzip=True)
    
'''
Returns the filename of the first file in the dataset
'''
def getDatasetFileName(datasetName):
    return str(kaggle.api.dataset_list_files(datasetName).files[0])

'''
Takes in a list of strings
    - Replaces URLs with 'http'
    - Replaces '@Username' with '@'
    - Removes unreadable characters

Returns a list with these changes    
'''
def cleanDataset(listToBeCleaned):
    cleanList = [re.sub(r'[^\x00-\x7F]+|amp', '',item).replace("'", '') for item in listToBeCleaned]
    cleanList = [re.sub(r'http\S+', 'http', item) for item in cleanList]
    cleanList = [re.sub(r'@\S+', '@', item) for item in cleanList]
    
    return cleanList

'''
Makes a dictionary that stores info for each class
    -Takes in the MNB obejct, feature names, and class names
    -You can also adjust the minimum count for an important word, that way you don't receive any outliers'
    -You can also change the number of top words you receive
'''
def constructResultsDictionary(mnbInput, featureNames, classNames=['brand','female','male'], minCount=3, numTopWords=10):
    genderResults = {className:{} for className in classNames}
    weights = mnbInput.coef_
    counts = mnbInput.feature_count_

    for index, name in enumerate(classNames):
        genderResults[name]['weights'] = weights[index].argsort()
        genderResults[name]['counts'] = counts[index].argsort()
        
    for name in genderResults:
        topWords = []
        for i in reversed(genderResults[name]['weights']):
            if genderResults[name]['counts'][i] > minCount and featureNames[i][0] != '_':
                topWords.append(featureNames[i])
            if len(topWords) > (numTopWords-1):
                genderResults[name]['topWords'] = topWords
                break
        
    return genderResults