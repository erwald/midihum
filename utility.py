import numpy as np


def exists(it):
    return (it is not None)


def map_removing_none(function, sequence):
    return filter(exists, map(function, sequence))


def replace_nan_with_average(array):
    '''Takes an array, and replaces all of the `nan` elements in it with the 
    average for that column.'''
    column_means = np.nanmean(array, axis=0)
    indices = np.where(np.isnan(array))
    array[indices] = np.take(column_means, indices[1])

    return array
