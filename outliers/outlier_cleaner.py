#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    size_dataset = len(predictions)
    ### create the list of errors
    errors = [(net_worths[i]-predictions[i]) ** 2 for i in range(size_dataset)]
    ### merge the three lists in cleaned_data list
    cleaned_data = zip(ages, net_worths, errors)
    ### remove 10% of the data having the greatest squared errors
    nb_to_remove = int(size_dataset*0.1)
    for i in range(nb_to_remove):
        index = cleaned_data.index(max(cleaned_data, key=lambda t: t[2]))
        cleaned_data.pop(index)

    return cleaned_data
