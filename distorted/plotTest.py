import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np


def plot_quantity(disorted_instance, column_name):
    """
    Generates an annular plot corresponding to a given quantity.
    """
    Radii = disorted_instance.df['Span'].to_numpy()
    Angles = disorted_instance.df['Theta'].to_numpy()
    Values = disorted_instance.df[column_name].to_numpy()
    x = np.array(Radii)*np.sin(Angles)
    y = np.array(Radii)*np.cos(Angles)
    Xi = np.linspace(-1,1,50)
    Yi = np.linspace(-1,1,50)

