#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)



from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]


### How many training points are there, according to the starter code?
print "There are", len(labels_train), "training points."

### What's the accuracy of the decision tree you just made?
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
tree = DecisionTreeClassifier()
tree.fit(features_train, labels_train)
pred = tree.predict(features_test)
print "The accuracy of the decision tree is", accuracy_score(labels_test, pred)

### What's the importance of the most important feature?
### What's the number of that feature?
imp = 0
nb = -1
i = 0
for fi in tree.feature_importances_:
    if fi > imp:
        imp = fi
        nb = i
    i += 1
print "Feature", nb, "is the feature the most important with an importance of", imp
### Which word correspond to that feature?
feature_names = vectorizer.get_feature_names()
print "It correspond to the word", feature_names[nb]
### Once you removed the signature words, which words have an importance greater than 0.2?
i = 0
for fi in tree.feature_importances_:
    if fi > 0.2:
        print "Word", feature_names[nb], "has an importance of", fi
    i += 1
