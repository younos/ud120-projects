#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
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
### print out the accuracy
pred = tree.predict(features_test)
from sklearn.metrics import accuracy_score
print 'The accuracy is', accuracy_score(labels_test, pred)
