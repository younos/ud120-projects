#!/usr/bin/python

"""
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)


### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### rescale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
finance_features = scaler.fit_transform(finance_features)
###  what would be the rescaled value of a 'salary' feature an initial value of $200,000?
### and an 'exercised_stock_options' of $1 million?
rescaled_sample = scaler.transform(numpy.array([[200000, 1000000]]))
print "A salary of 200'000 would have a rescaled value of", rescaled_sample[0][0]
print "An exercised_stock_options of 1'000'000 would have a rescaled value of", rescaled_sample[0][1]


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(finance_features)
pred = kmeans.labels_

### what are the maximum and minimum values taken by the "exercised_stock_options" feature?
eso_list = [val['exercised_stock_options'] for val in data_dict.values() if val['exercised_stock_options'] != 'NaN']
print 'Maximum of exercised_stock_options is', max(eso_list)
print 'Minimum of exercised_stock_options is', min(eso_list)
### what are the maximum and minimum values taken by the "salary" feature?
sal_list = [val['salary'] for val in data_dict.values() if val['salary'] != 'NaN']
print 'Maximum of salary is', max(sal_list)
print 'Minimum of salary is', min(sal_list)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
