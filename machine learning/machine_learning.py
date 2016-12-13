import pandas as pd
import quandl, math, datetime, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

# volume = how many trades occured that day
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# High-Low Percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.00
df['PCT_change'] = (df['Adj. Close'] + df['Adj. Open']) / df['Adj. Open'] * 100.00

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# math.ceil (teto) Ex: len(df) = 10.2. So, it will turn it into 11. (rounds everything up)
# 0.01 - we will want to predict the 1% of the data frame
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)

# -------------------------- Testing with 30 days of advance  -----------------------------------
'''
# X - features   |   y - labels
# The features will be everything excepet the label column
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# Now we will scale X, before we fit the data.
X = preprocessing.scale(X)
y = np.array(df['label'])

# We will use 20% of the data as test data.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Fit our classifiers
clf = LinearRegression()
clf.fit(X_train, y_train)

# Testing
accuracy = clf.score(X_test, y_test)

# The linear regression showed a good result(96% accuracy), but when using the SVR the result was not so good(55%, 68%)
'''

# -------------------------- Predicting the future (next 30 days)  -----------------------------------

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#clf = LinearRegression()
#clf.fit(X_train, y_train)

# Here we save the classifier, so we can avoid training it again and again everytime the script runs
#with open('linearregression.pickle', 'wb') as file:  #wb - write bytes
#	pickle.dump(clf, file)

# Doing this we can comment the above lines after saving it for the first time 
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

# Predicting the future (next 30 days)
forecast_set = clf.predict(X_lately)

print forecast_set, accuracy, forecast_out

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
#last_unix = last_date.timestamp()
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

# loop through the forecast_set making the feature features not a number (nan). It will be used to set Date as X label
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()