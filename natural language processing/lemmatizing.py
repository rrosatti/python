from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# the default parameter for 'pos' is noun

print(lemmatizer.lemmatize("cats"))
# a - adjetive
print(lemmatizer.lemmatize("better", pos='a'))
print(lemmatizer.lemmatize("best", pos='a'))

print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("ran", pos='v'))