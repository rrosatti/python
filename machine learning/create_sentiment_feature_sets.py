'''
Ex: lexicon: [chair, table, spoon, television]

Sentence: I pulled the chair up to the table

Converting this sentence to a vector: [1,1,0,0]
'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg):
	# populate the lexicon with the words in the files pos.txt and neg.txt
	lexicon = []
	for fi in [pos,neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)

	# l2 is the final lexicon
	# We don't want common words (the, and, of)
	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)

	print(len(l2))


	return l2


# classifying features sets
def sample_handling(sample, lexicon, classification):
	# featureset will be a list of lists
	featureset = []

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))

			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			featureset.append([features, classification])

	return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1,0])
	features += sample_handling('neg.txt', lexicon, [0,1])
	random.shuffle(features)

	features = np.array(features)

	# default = 10% of the data
	test_size = int(test_size*len(features))

	'''
	What [:,0] means:

	Ex: [[5,8], [7,9]]
	Res: [5,7]

	So, it will take all the elements in the 0s index

	In our case it will be:
	[[[0,1,0,1,1]], [0,1]], ...]
	[[features, label]]

	'''
	train_x = list(features[:,0][:-test_size])
	train_y = list(features[:,1][:-test_size])

	test_x = list(features[:,0][-test_size:])
	test_y = list(features[:,1][-test_size:])

	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt', test_size=0.1)
	with open('C:/Users/rodri/OneDrive/Documentos/python/data/sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)