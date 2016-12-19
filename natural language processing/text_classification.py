import nltk, random, pickle
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/nlp_data/'

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		# count how many occurrences of the most popular vote was in the list
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf		

def save_pickle_data(data, filename):
	with open(data_path+filename, 'wb') as f:
		pickle.dump(data, f)

def get_pickle_data(filename):
	with open(data_path+filename, 'rb') as f:
		data = pickle.load(f)
	return data

# the commented code refers to the old data set
'''
# it will be a list of tuples
documents = [(list(movie_reviews.words(fileid)), category) 
				for category in movie_reviews.categories() 
				for fileid in movie_reviews.fileids(category)]

'''
# another way of doing the same thing
'''
documents = []
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		documents.append((list(movie_reviews.words(fileid)), category))
'''
'''
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())
'''

with open(data_path+'positive.txt', 'r') as f:
	short_pos = f.read()	

with open(data_path+'negative.txt', 'r') as f:
	short_neg = f.read()

documents = []
all_words = []

# j is adjective, r is adverb, and v is verb
# allowerd_word_types = ['J','R','V']
allowed_word_types = ['J']

for p in short_pos.split('\n'):
	# list of tuples (review, pos/neg)
	documents.append( (p, 'pos') )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

for p in short_neg.split('\n'):
	documents.append( (p, 'neg') )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

save_pickle_data(documents,'documents.pickle')

# arange the words (most common words to the least common words)
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words['awesome'])

# top 5000 words (we used 3000 words in the first data set)
word_features = list(all_words.keys())[:5000]

save_pickle_data(word_features,'word_features5k.pickle')

def find_features(document):
	#words = set(document) # used in first data set
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		# True if the word is in the doc or False if not
		features[w] = (w in words)

	return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_pickle_data(featuresets, 'featuresets.pickle')

# doing this we prevent the data of being only positive or only negative
random.shuffle(featuresets)

# we used 1900 in the first dataset
training_set = featuresets[:10000]
testing_set = featuresets[10000:]


# posterior = prior occurences x likelihood / evidence

# create the classifier (run it only once, then 'pickle it')
clf = nltk.NaiveBayesClassifier.train(training_set)
save_pickle_data(clf, 'originalnaivebayes5k.pickle')

#clf = get_pickle_data('posnegclf.pickle')
#print('Original Naive Bayes Algo accuracy percent:', (nltk.classify.accuracy(clf, testing_set))*100)
# show the most (15) informative features
#clf.show_most_informative_features(15)




# using the Multinomial Naive Bayes
MNB_clf = SklearnClassifier(MultinomialNB())
# train classifier
MNB_clf.train(training_set)
print('MNB_clf accuracy percent:', (nltk.classify.accuracy(MNB_clf, testing_set))*100)

save_pickle_data(MNB_clf, 'MNB_clf5k.pickle')

# using the Gaussian Naive Bayes
#GNB_clf = SklearnClassifier(GaussianNB())
#GNB_clf.train(training_set)
#print('GNB_clf accuracy percent:', (nltk.classify.accuracy(GNB_clf, testing_set))*100)

# using the Bernoulli Naive Bayes
BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print('BNB_clf accuracy percent:', (nltk.classify.accuracy(BNB_clf, testing_set))*100)

save_pickle_data(BNB_clf, 'BNB_clf5k.pickle')

# using the Logistic Regression
LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print('Logistic Regression accuracy percent:', (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

save_pickle_data(LogReg_clf, 'LogReg_clf5k.pickle')

# using the Stochastic Gradient Decent Classifier
SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print('Stochastic Gradient Decent Classifier accuracy percent:', (nltk.classify.accuracy(SGD_clf, testing_set))*100)

save_pickle_data(SGD_clf, 'SGD_clf5k.pickle')

# using the C-Support Vector Classification
# it gave us inaccurate numbers, so we won't use it
#SVC_clf = SklearnClassifier(SVC())
#SVC_clf.train(training_set)
#print('Support Vector Classification accuracy percent:', (nltk.classify.accuracy(SVC_clf, testing_set))*100)

# using the Linear Support Vector Classification
LSVC_clf = SklearnClassifier(LinearSVC())
LSVC_clf.train(training_set)
print('Linear Support Vector Classification accuracy percent:', (nltk.classify.accuracy(LSVC_clf, testing_set))*100)

save_pickle_data(LSVC_clf, 'LSVC_clf5k.pickle')

# using the Nu(Number) Support Vector Classification
NSVC_clf = SklearnClassifier(NuSVC())
NSVC_clf.train(training_set)
print('Nu(Number) Support Vector Classification accuracy percent:', (nltk.classify.accuracy(NSVC_clf, testing_set))*100)

save_pickle_data(NSVC_clf, 'NSVC_clf5k.pickle')
# here we removed the 'clf' and 'SGD_clf' (inaccurate values)
voted_clf = VoteClassifier(MNB_clf, BNB_clf, LogReg_clf, LSVC_clf, NSVC_clf)
print('Voted Classifier accuracy percent:', (nltk.classify.accuracy(voted_clf, testing_set))*100)

print('Classification:', voted_clf.classify(testing_set[0][0]), "Confidence %:", voted_clf.confidence(testing_set[0][0])*100)
print('Classification:', voted_clf.classify(testing_set[1][0]), "Confidence %:", voted_clf.confidence(testing_set[1][0])*100)
print('Classification:', voted_clf.classify(testing_set[2][0]), "Confidence %:", voted_clf.confidence(testing_set[2][0])*100)
print('Classification:', voted_clf.classify(testing_set[3][0]), "Confidence %:", voted_clf.confidence(testing_set[3][0])*100)
print('Classification:', voted_clf.classify(testing_set[4][0]), "Confidence %:", voted_clf.confidence(testing_set[4][0])*100)
print('Classification:', voted_clf.classify(testing_set[5][0]), "Confidence %:", voted_clf.confidence(testing_set[5][0])*100)


def sentiment(text):
	feats = find_features(text)
	return voted_clf.classify(feats)