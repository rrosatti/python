import nltk, random, pickle
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

documents = get_pickle_data('documents.pickle')

word_features = get_pickle_data('word_features5k.pickle')

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		# True if the word is in the doc or False if not
		features[w] = (w in words)

	return features

featuresets = get_pickle_data('featuresets.pickle')

# doing this we prevent the data of being only positive or only negative
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]


clf = get_pickle_data('originalnaivebayes5k.pickle')

# using the Multinomial Naive Bayes
MNB_clf = get_pickle_data('MNB_clf5k.pickle')

# using the Bernoulli Naive Bayes
BNB_clf = get_pickle_data('BNB_clf5k.pickle')

# using the Logistic Regression
LogReg_clf = get_pickle_data('LogReg_clf5k.pickle')

# using the Stochastic Gradient Decent Classifier
SGD_clf = get_pickle_data('SGD_clf5k.pickle')

# using the Linear Support Vector Classification
LSVC_clf = get_pickle_data('LSVC_clf5k.pickle')

# using the Nu(Number) Support Vector Classification
NSVC_clf = get_pickle_data('NSVC_clf5k.pickle')

# here we removed the 'SGD_clf' (inaccurate values)
voted_clf = VoteClassifier(clf, MNB_clf, BNB_clf, LogReg_clf, LSVC_clf, NSVC_clf)

def sentiment(text):
	feats = find_features(text)
	return voted_clf.classify(feats), voted_clf.confidence(feats)