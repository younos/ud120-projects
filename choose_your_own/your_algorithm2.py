#!/usr/bin/python

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the RF classifier with custom parameters and train it
clf = RandomForestClassifier(n_estimators=10, min_samples_split=40)
clf.fit(features_train, labels_train)

# Get the prediction of the features to test
pred = clf.predict(features_test)

# Compute the accuracy
accuracy = accuracy_score(labels_test, pred)
print "accuracy is:", accuracy

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
