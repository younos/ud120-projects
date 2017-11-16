#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### use the train_test_split validation available in sklearn.cross_validation
### hold out 30% of the data for testing and set the random_state parameter to 42
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### create a decision tree classifier (just use the default parameters)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
### train it on the data
tree.fit(features_train, labels_train)
### get the predictions of the test set
pred = tree.predict(features_test)
### how many POIs are predicted for the test set for your POI identifier?
nb_poi = sum(pred)
print 'There are', int(nb_poi), 'predicted POIs in the test set.'
### how many people total are in your test set?
nb_peo = len(pred)
print 'There are', nb_peo, 'people in the test set.'
### if your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
from sklearn.metrics import accuracy_score
print 'The accuracy would be', accuracy_score(labels_test, [0.]*nb_peo), 'if the identifier predicted 0 for everyone.'
### do you get any true positives with your DT?
print 'There is/are', int(sum([t*p for t, p in zip(labels_test, pred)])), 'true positives.'
### What's the precision?
from sklearn.metrics import precision_score
print 'The precision is', precision_score(labels_test, pred)
### What's the recall?
from sklearn.metrics import recall_score
print 'The recall is', recall_score(labels_test, pred)
