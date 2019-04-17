from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from FeatureExtraction import *

import json
import numpy as np
import csv

def split_data(data,labels):
    return train_test_split(data, labels, test_size=0.2, random_state=42, 
        shuffle="true")


def train_NB(train_data, train_labels):
    return MultinomialNB().fit(train_data, train_labels)


def train_random_foest(train_data, train_labels, est):
    return RandomForestClassifier(n_estimators=est).fit(train_data, 
        train_labels)


def test_classifier(clf, validate_data, validate_labels, str):
    predicted = clf.predict(validate_data)
    print(str)
    print(np.mean(predicted == validate_labels))

def get_News_dataset():
	ml_data = list()
	ml_labels = list()
	with open("News_dataset/Fake.csv") as csv_file:
	    csv_reader = csv.reader(csv_file, delimiter=',')
	    for row in csv_reader:
	        ml_data.append(row[1])
	        ml_labels.append(0)
	with open("News_dataset/CleanTrue.csv") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			ml_data.append(row[1])
			ml_labels.append(1)
	return ml_data, ml_labels

def get_FNN():
    ml_data = list()
    ml_labels = list()
    # open News.txt
    with open("FakeNewsNet/News.txt") as f:
        # for each line in News.txt
        for line in f:
            # read in the data (ie filename)

            # create openable file name
            json_filename = "FakeNewsNet/"+line.rstrip()+"-Webpage.json"
            
            # open file and read everything 
            with open(json_filename, encoding='utf-8') as data_file:
                data = json.loads(data_file.read())
                
                # create data array
                ml_data.append(data['text'])
                
                # create label array
                if "Real" in json_filename:
                    ml_labels.append(1)
                else:
                    ml_labels.append(0)
    return ml_data, ml_labels

def get_OriNews():
    ml_data = list()
    ml_labels = list()
    with open("MyNews/researcharticles.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            filename = "MyNews/" + row[0]
            if row[3] == "Not-Real-Other":
                with open(filename, encoding='utf-8') as data_file:
                    ml_data.append(data_file.read())
                    ml_labels.append(0)
            elif row[3] == "Real":
                with open(filename, encoding='utf-8') as data_file:
                    ml_data.append(data_file.read())
                    ml_labels.append(1)
    return ml_data, ml_labels