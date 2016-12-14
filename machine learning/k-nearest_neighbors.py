# For more information about the data, check it out: http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
# the -99999 is used in order to not lose all the 'compromised' data signed with '?' sign. So, this particular information will be treated
# as an outlier
df.replace('?', -99999, inplace=True)

# drop useless information
df.drop(['id'], 1, inplace=True)

# X = features | y = labels or class
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# do the cross validation
# 20% of the data will be used for testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

# making a prediction
example_measures = np.array([4,2,1,1,1,2,3,2,1])
prediction = clf.predict(example_measures)

example_measures_two_samples = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example_measures_two_samples = example_measures_two_samples.reshape(len(example_measures_two_samples), -1)

prediction_two_samples = clf.predict(example_measures_two_samples)

print prediction, prediction_two_samples