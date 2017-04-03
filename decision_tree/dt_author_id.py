#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

# Get number of features
print len(features_train[0])

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize the classifier
clf = DecisionTreeClassifier(min_samples_split=40)

# Train the classifier and measure the time
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# Get the prediction of the features to test and measure the time
t0 = time()
pred = clf.predict(features_test)
print "testing time:", round(time()-t0, 3), "s"

# Get the accuracy
accuracy = accuracy_score(labels_test, pred)
print accuracy

#########################################################
