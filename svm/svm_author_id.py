#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the classifier
clf = SVC(kernel='rbf', C=10000.0)

# Reduce the training set to 1%
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

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
# Get the prediction for testing samples 10, 26 and 50
print "prediction of testing sample no 10 is " + str(pred[10])
print "prediction of testing sample no 26 is " + str(pred[26])
print "prediction of testing sample no 50 is " + str(pred[50])
# Get the number of predictions for each class
chris = sum(pred)
print "there are " + str(len(pred)-chris) + " events predicted as Sara"
print "there are " + str(chris) + " events predicted as Chris"

#########################################################
