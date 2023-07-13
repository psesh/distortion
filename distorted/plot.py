import matplotlib.pyplot as plt
import numpy
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import Rbf

def plot_quantity(distorted_instance, column_name):
    """
    Generates an annular plot corresponding to a given quantity.
    """
    Radii = distorted_instance.df['Span']
    Angles = distorted_instance.df['Theta']
    Values = distorted_instance.df[column_name]

    minSpan = np.min(distorted_instance.df['Span'])
    minSpanValues = distorted_instance.df.loc[distorted_instance.df['Span'] == minSpan, column_name]
    avgMinSpanValue = np.average(minSpanValues)
    centerValue = avgMinSpanValue
    x = np.array(Radii)*np.sin(Angles)
    y = np.array(Radii)*np.cos(Angles)

    x = numpy.insert(x,0,0)
    y = numpy.insert(y,0,0)
    Values = numpy.insert(Values,0,centerValue)

    extent = np.max([np.abs(np.min(x)),np.abs(np.max(x)),np.abs(np.min(y)),np.abs(np.max(y))])

    xi, yi = np.mgrid[-extent:extent:500j, -extent:extent:500j]
    Vi = griddata((x, y), Values, (xi, yi), method='linear')

    f = plt.figure(figsize=(9,6))
    ax = f.add_subplot(111)
    pax = f.add_subplot(111, polar=True, facecolor = 'None')
    pax.set_ylim([0, extent])
    ax.set_aspect(1)
    pax.set_aspect(1)
    ax.axis('Off')
    cf = ax.contourf(xi,yi,Vi, 60, cmap=plt.cm.jet)
    ax.scatter(x, y,color = 'black', s = 5)
    cbar_ax = f.add_axes([0.85, .1, 0.05,.8])
    plt.colorbar(cf, cax = cbar_ax)
    plt.show()
