import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np 

def plot_quantity(disorted_instance, column_name):
    """
    Generates an annular plot corresponding to a given quantity.
    """
    Radii = disorted_instance.df['Span']
    Angles = disorted_instance.df['Theta']
    Values = disorted_instance.df[column_name]
    x = np.array(Radii)*np.sin(Angles)
    y = np.array(Radii)*np.cos(Angles)
    Xi = np.linspace(-1,1,50)
    Yi = np.linspace(-1,1,50)

    f = plt.figure()
    left, bottom, width, height= [0,0, 1, 0.7]
    ax  = plt.axes([left, bottom, width, height])
    pax = plt.axes([left, bottom, width, height], 
                    projection='polar')
    cax = plt.axes([0.8, 0, 0.05, 1])
    ax.set_aspect(1)
    ax.axis('Off')
    Vi = griddata((x, y), Values, (Xi[None,:], Yi[:,None]), method='linear')
    cf = ax.contourf(Xi,Yi,Vi, 30, cmap=plt.cm.jet)
    plt.show() 