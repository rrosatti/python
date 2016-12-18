import nltk, random, pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/'

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


# it will be a list of tuples
documents = [(list(movie_reviews.words(fileid)), category) 
				for category in movie_reviews.categories() 
				for fileid in movie_reviews.fileids(category)]

'''
# another way of doing the same thing

documents = []
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		documents.append((list(movie_reviews.words(fileid)), category))
'''

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())

# arange the words (most common words to the least common words)
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words['awesome'])

# top 3000 words
word_features = list(all_words.keys())[:3000]

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		# True if the word is in the doc or False if not
		features[w] = (w in words)

	return features

def save_pickle_data(data, filename):
	with open(data_path+'filename', 'wb') as f:
		pickle.dump(data, f)

def get_pickle_data(filename):
	with open(data_path+'filename', 'rb') as f:
		data = pickle.load(f)
	return data

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]


# posterior = prior occurences x likelihood / evidence

# create the classifier (run it only once, then 'pickle it')
#clf = nltk.NaiveBayesClassifier.train(training_set)

clf = get_pickle_data('naivebayes.pickle')
print('Original Naive Bayes Algo accuracy percent:', (nltk.classify.accuracy(clf, testing_set))*100)
# show the most (15) informative features
clf.show_most_informative_features(15)

#save_pickle_data(clf, 'naivebayes.pickle')



# using the Multinomial Naive Bayes
MNB_clf = SklearnClassifier(MultinomialNB())
# train classifier
MNB_clf.train(training_set)
print('MNB_clf accuracy percent:', (nltk.classify.accuracy(MNB_clf, testing_set))*100)

# using the Gaussian Naive Bayes
#GNB_clf = SklearnClassifier(GaussianNB())
#GNB_clf.train(training_set)
#print('GNB_clf accuracy percent:', (nltk.classify.accuracy(GNB_clf, testing_set))*100)

# using the Bernoulli Naive Bayes
BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print('BNB_clf accuracy percent:', (nltk.classify.accuracy(BNB_clf, testing_set))*100)

# using the Logistic Regression
LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print('Logistic Regression accuracy percent:', (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

# using the Stochastic Gradient Decent Classifier
SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print('Stochastic Gradient Decent Classifier accuracy percent:', (nltk.classify.accuracy(SGD_clf, testing_set))*100)

# using the C-Support Vector Classification
# it gave us inaccurate numbers, so we won't use it
#SVC_clf = SklearnClassifier(SVC())
#SVC_clf.train(training_set)
#print('Support Vector Classification accuracy percent:', (nltk.classify.accuracy(SVC_clf, testing_set))*100)

# using the Linear Support Vector Classification
LSVC_clf = SklearnClassifier(LinearSVC())
LSVC_clf.train(training_set)
print('Linear Support Vector Classification accuracy percent:', (nltk.classify.accuracy(LSVC_clf, testing_set))*100)

# using the Nu(Number) Support Vector Classification
NSVC_clf = SklearnClassifier(NuSVC())
NSVC_clf.train(training_set)
print('Nu(Number) Support Vector Classification accuracy percent:', (nltk.classify.accuracy(NSVC_clf, testing_set))*100)


voted_clf = VoteClassifier(clf, MNB_clf, BNB_clf, LogReg_clf, SGD_clf, LSVC_clf, NSVC_clf)
print('Voted Classifier accuracy percent:', (nltk.classify.accuracy(voted_clf, testing_set))*100)

print('Classification:', voted_clf.classify(testing_set[0][0]), "Confidence %:", voted_clf.confidence(testing_set[0][0])*100)
print('Classification:', voted_clf.classify(testing_set[1][0]), "Confidence %:", voted_clf.confidence(testing_set[1][0])*100)
print('Classification:', voted_clf.classify(testing_set[2][0]), "Confidence %:", voted_clf.confidence(testing_set[2][0])*100)
print('Classification:', voted_clf.classify(testing_set[3][0]), "Confidence %:", voted_clf.confidence(testing_set[3][0])*100)
print('Classification:', voted_clf.classify(testing_set[4][0]), "Confidence %:", voted_clf.confidence(testing_set[4][0])*100)
print('Classification:', voted_clf.classify(testing_set[5][0]), "Confidence %:", voted_clf.confidence(testing_set[5][0])*100)
