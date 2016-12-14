'''
Eucledian Distance
Ex:

q = (1, 3)
P = (2, 5)

sqrt( (1 - 2)^2 + (3 - 5)^2 )

'''
from __future__ import division
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

# -------------------------  Simple Data --------------------------------

# it will be two-dimensional features
# We have here two classes(k and r) and their features
dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,6]]}
new_features = [5,7]

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups!')

	distances = []
	for group in data:
		for features in data[group]:
			eucledian_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([eucledian_distance, group])

	# here we will take the 'closest' distance
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	#print(Counter(votes).most_common(1))
	confidence = Counter(votes).most_common(1)[0][0] / k

	return vote_result, confidence

#result = k_nearest_neighbors(dataset, new_features, k=3)
#print result

#[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1], s=100, color=result)
#plt.show()


# -------------------------  Comparing our result with sklearn (real dataset) --------------------------------

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# convert it to a list of lists
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

# setting the test data as 20% of all data (the last 20%)
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# populate dictionaries
for i in train_data:
	# i[-1] corresponds to the last column (class: 2 or 4). So, we are appending a list of values to it
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbors(train_set, data, k=5)
		# testing the 'correctness' of our algorithm
		if group == vote:
			correct += 1
		else:
			print confidence
		total += 1

print "Accuracy: ", correct/total
