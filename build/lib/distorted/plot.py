import matplotlib.pyplot as plt
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

    x = np.array(Radii) * np.sin(Angles)
    y = np.array(Radii) * np.cos(Angles)

    extent = np.max([np.abs(np.min(x)), np.abs(np.max(x)), np.abs(np.min(y)), np.abs(np.max(y))])

    xi, yi = np.mgrid[-extent:extent:500j, -extent:extent:500j]

    Vi = griddata((x, y), Values, (xi, yi), method='linear')

    vEdge = Vi.flatten()
    xEdge = xi.flatten()
    yEdge = yi.flatten()
    xEdge = np.delete(xEdge, np.argwhere(np.isnan(vEdge)))
    yEdge = np.delete(yEdge, np.argwhere(np.isnan(vEdge)))
    vEdge = np.delete(vEdge, np.argwhere(np.isnan(vEdge)))

    V0 = griddata((xEdge, yEdge), vEdge, (xi, yi), method='nearest')

    Vi[np.isnan(Vi)] = V0[np.isnan(Vi)]

    midr, midc = Vi.shape[0] / 2, Vi.shape[1] / 2
    for er in range(Vi.shape[0]):
        for ec in range(Vi.shape[1]):
            if np.abs(np.sqrt((er - midr) ** 2 + (ec - midc) ** 2)) > Vi.shape[0] / 2:
                Vi[er][ec] = np.nan
            pass
        pass

    f = plt.figure(figsize=(9, 6))
    ax = f.add_subplot(111)
    pax = f.add_subplot(111, polar=True, facecolor='None')
    pax.set_ylim([0, extent])
    ax.set_aspect(1)
    pax.set_aspect(1)
    ax.axis('Off')
    cf = ax.contourf(xi, yi, Vi, 60, cmap=plt.cm.jet)
    ax.scatter(x, y, color='black', s=2)
    cbar_ax = f.add_axes([0.85, .1, 0.05, .8])
    plt.colorbar(cf, cax=cbar_ax)
    plt.show()
