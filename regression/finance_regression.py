#!/usr/bin/python

"""
    Starter code for the regression mini-project.

    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

from sklearn.linear_model import LinearRegression
### train the regression model with training data
reg1 = LinearRegression()
reg1.fit(feature_train, target_train)
### train the regression model with testing data
reg2 = LinearRegression()
reg2.fit(feature_test, target_test)

### extract the slope and the intercept
print 'Reg1: the slope is', reg1.coef_[0], 'and the intercept is', reg1.intercept_
print 'Reg2: the slope is', reg2.coef_[0], 'and the intercept is', reg2.intercept_

### get the R-squared value for training set and testing set
print 'Reg1: R-squared for training set is', reg1.score(feature_train, target_train)
print 'Reg1: R-squared for testing set is', reg1.score(feature_test, target_test)


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color )
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color )

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg1.predict(feature_test), color=train_color )
    plt.plot( feature_train, reg2.predict(feature_train), color=test_color )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
