import quandl, json, pickle, requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from statistics import mean
from sklearn import svm, preprocessing, cross_validation


'''
svm
preprocessing - help improve machine learning classification
cross validation - it will be used to create test and training sets
'''

style.use('fivethirtyeight')

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/'

def save_pickle_data(data, pickle_file):
	with open(data_path+pickle_file, 'wb') as f:
		pickle.dump(data, f)

def get_pickle_data(pickle_file):
	with open(data_path+pickle_file, 'rb') as f:
		data = pickle.load(f)
	return data

# If Future HPi drops, so it will labeled as a 0. And 1 if it goes up
def create_labels(cur_hpi, fut_hpi):
	if fut_hpi > cur_hpi:
		return 1
	else:
		return 0

housing_data = get_pickle_data('US_HPI.pickle')

housing_data = housing_data.pct_change()

housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_data['US_HPI_Future'] = housing_data['United States'].shift(-1)
housing_data.dropna(inplace=True)

housing_data['label'] = list(map(create_labels, housing_data['United States'], housing_data['US_HPI_Future']))



print(housing_data.head())

# X - features | y - labels
X = np.array(housing_data.drop(['label', 'US_HPI_Future'], 1))
# it converts the data to an arrange of positive 1s and negative 1s
X = preprocessing.scale(X)
y = np.array(housing_data['label'])

# the test size will be 20% of all data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# train the classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))