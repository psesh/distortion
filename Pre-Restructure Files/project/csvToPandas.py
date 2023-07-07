"""
csvToPanda.py
=============
Converts a csv of rake total pressure data into a pandas dataframe and sorts it for other functions.
.csv should include header with column order [Rake Number, Span, Theta (rad), Pressure (units)]
"""

import pandas as pd


def csvToPandas(filename):
    """Creates pandas dataframe from csv pressure rake data
    .csv must have header with column order [Rake Number, Span, Theta (rad), Pressure (units)]

    :param filename: Name of .csv file
    :type filename: string
    :return: Sorted pandas dataframe of pressure rake data
    :rtype: Pandas dataframe
    """
    if filename[-4:] != '.csv':
        filename = filename + '.csv'

    data = pd.read_csv(filename)

    column_names = list(data.columns)
    data = data.sort_values([column_names[2], column_names[1]])

    return data
