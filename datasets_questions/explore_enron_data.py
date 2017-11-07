#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
from math import isnan

# Import dictionary objet stored in a pkl file
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


# QUESTIONS

# How many data points (people) are in the dataset?
nb_people = len(enron_data)
print 'There is', nb_people, 'people in the dataset'

# For each person, how many features are available?
print 'There is', len(enron_data["SKILLING JEFFREY K"]), 'features in the dataset'

# How many POIs are there in the E+F dataset?
nb_poi = sum([1 for val in enron_data.values() if val['poi'] == 1])
print 'There is', nb_poi, 'persons of interest in the dataset'

# What is the total value of the stock belonging to James Prentice?
print 'James Prentice has', enron_data['PRENTICE JAMES']['total_stock_value'], 'of stock value'

# How many email messages do we have from Wesley Colwell to persons of interest?
print 'Wesley Colwell sent', enron_data['COLWELL WESLEY']['from_this_person_to_poi'], 'e-mails to POIs'

# What's the value of stock options exercised by Jeffrey K Skilling?
print "Jeffrey K Skilling's value of stock options exercised is", enron_data['SKILLING JEFFREY K']['exercised_stock_options']

# Among Lay, Skilling and Fastow, who took home the most money?
payments_dic = {'Jeffrey K Skilling': enron_data['SKILLING JEFFREY K']['total_payments'],
                'Kenneth Lay': enron_data['LAY KENNETH L']['total_payments'],
                'Andrew Fastow': enron_data['FASTOW ANDREW S']['total_payments']}
person = max(payments_dic.iterkeys(), key=(lambda key: payments_dic[key]))
print 'The person who took home the most money is', person, "with", payments_dic[person]

# How many people have a salary in the dataset? How about e-mail address?
nb_sal = sum([1 for val in enron_data.values() if not isnan(float(val['salary']))])
print nb_sal, 'persons have a salary in the dataset'
nb_add = sum([1 for val in enron_data.values() if val['email_address'] != 'NaN'])
print nb_add, 'persons have an e-mail address in the dataset'

# How many people in the E+F dataset have "NaN" for their total payments?
# What percentage of people in the dataset as a whole is this?
nb_nopay = sum([1 for val in enron_data.values() if val['total_payments'] == 'NaN'])
print nb_nopay, 'persons have "NaN" for their total payments, which means', '{:.2f}%'.format(float(nb_nopay)/nb_people * 100)

# How many of POIs in the E+F dataset have "NaN" for their total payments?
# What percentage of POIs in the dataset as a whole is this?
nb_nopaypoi = sum([1 for val in enron_data.values() if val['total_payments'] == 'NaN' and val['poi'] == 1])
print nb_nopaypoi, 'POIs have "NaN" for their total payments, which means', '{:.2f}%'.format(float(nb_nopaypoi)/nb_poi * 100), 'among the POIs'

# If you added in 10 more data points which were all POIs, and put "NaN" for the total payments for those folks.
# What is the new number of people of the dataset?
print 'There is', nb_people+10, 'people in the dataset'
# What is the new number of folks with "NaN" for total payments?
print 'There is', nb_nopay+10, 'people in the dataset having "NaN" for their total payments'
# What is the new number of POIs in the dataset?
print 'There is', nb_poi+10, 'POIs in the dataset'
# What is the new number of POIs with NaN for total_payments?
print nb_nopaypoi+10, 'POIs have "NaN" for their total payments, which means', '{:.2f}%'.format(float(nb_nopaypoi+10)/(nb_poi+10) * 100), 'among the POIs'
