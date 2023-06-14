"""
averagingMethods.py
======================
Functions and Script to calculate various averages from pressure rake data
"""
import pandas as pd
import numpy as np


def areaWeightedAvg(filename, hubRadius, casingRadius):
    """Returns a single area weighted average total pressure from rake data
    Each probe is assigned an annular sector, whose extents defined by the angles and radii midway between adjacent
    probes. Average pressure is found by weighing each probe pressure by their assigned annular sectors.

    :math:`A = \sum_{i=1}^{N} p_i \gamma_i` where :math:`p_i` are individual measurements and :math:`\gamma_i` are the associated weights.
        
    :param filename: The name of the .csv file containing pressure probe data. Column Format: [Probe Number, Span, Theta(radians), Pressure). Must be in same folder as areaWeightedAverage.py
    :type filename: string
    :param hubRadius: Radius of hub outer edge
    :type hubRadius: float
    :param casingRadius: Radius of casing inner edge
    :type casingRadius: float
    :return: A single averaged pressure value
    :rtype: float
    """

    # The lines below should be repalced with pandas directly, i.e.,
    df = pd.read_csv(filename)
    # etc
    
    #if filename[-4:] != '.csv':
    #    filename = filename + '.csv'
    #data = np.genfromtxt(filename, delimiter=',')
    #data = np.delete(data, 0, axis=0)

    #data = data[data[:, 1].argsort()]  # sorting by span
    #data = data[data[:, 2].argsort(kind='mergesort')]  # sorting by theta

    thetasRaw = data[:, 2]  # Raw list of thetas from sorted csv (has repeats)
    diff_list = np.diff(thetasRaw)  # Finding differences between adjacent theta values
    I = np.nonzero(diff_list)  # Determining the location of first nonzero element
    probesOnRake = 1 + I[0][0]  # Number of probes on each rake
    thetas = thetasRaw[::probesOnRake]  # Creating new thetas array with no repeats
    thetaDifferences = np.diff(thetas) / 2  # Half the angle between each rake
    thetaDifferences = np.concatenate((thetaDifferences, [((thetas[0] + (2 * np.pi - thetas[-1])) / 2)]), axis=0)  #
    # Creating list of differences in angle between adjacent rakes, divided by 2
    sectionEdgeAngles = thetas + thetaDifferences  # Creating list of angles midway between each rake - rake bisectors
    sectionDiff = np.diff(sectionEdgeAngles)  # Finding the difference in angle between rake bisectors
    sectorAngles = sectionEdgeAngles[0] - sectionEdgeAngles[-1] % (2 * np.pi)  # Assigning first annular sector angle
    # since it wraps around array
    sectorAngles = np.concatenate(([sectorAngles], sectionDiff), axis=0)  # Adding rest of sector angles
    # TODO There's almost certainly a better way of doing some of this angle math ^

    spans = data[0:probesOnRake, 1]  # Spans of probes on one rake
    spanDifferences = np.diff(spans) / 2  # Half the distance between each probe radially
    circleRadii = np.concatenate(([hubRadius], spans[0:-1] + spanDifferences, [casingRadius]), axis=0)  # Combining
    # hubRadius, casingRadius, and midpoints of probes to list relevant circle radii
    ringAreas = []
    for i, radius in enumerate(circleRadii):
        if i < len(circleRadii) - 1:
            area = np.pi * (pow(circleRadii[i + 1], 2) - pow(circleRadii[i], 2))
            ringAreas = ringAreas + [area]
    # List of the relevant ring areas from center outwards

    effectiveArea = sum(ringAreas)  # Sum of rings to find effective area (area between casing and hub)
    weightedAverage = 0
    for i, pressure in enumerate(data[:, 3]):
        ringNumber = (i + 1) % len(ringAreas) - 1
        angleNumber = (i) // len(ringAreas)
        term = (pressure * ringAreas[ringNumber] * (sectorAngles[angleNumber] / (2 * np.pi))) / effectiveArea
        weightedAverage += term
    # Multiplying fractional area by that sector's pressure, and summing to find complete weighted average

    return weightedAverage


def radialAverage(filename):
    """Returns radially averaged total pressures from rake data. (One averaged value per angle of probes)

    :param filename: The name of the .csv file containing pressure probe data. Column Format: [Probe Number, Span, Theta(radians), Pressure). Must be in same folder as areaWeightedAverage.py
    :type filename: string
    :return: Array of average pressures for each theta value (in order of increasing theta)
    :rtype: float array
    """

    if filename[-4:] != '.csv':
        filename = filename + '.csv'
    data = np.genfromtxt(filename, delimiter=',')
    data = np.delete(data, 0, axis=0)

    data = data[data[:, 1].argsort()]  # sorting by span
    data = data[data[:, 2].argsort(kind='mergesort')]  # sorting by theta

    thetasRaw = data[:, 2]  # Raw list of thetas from sorted csv (has repeats)
    diff_list = np.diff(thetasRaw)  # Finding differences between adjacent theta values
    I = np.nonzero(diff_list)  # Determining the location of first nonzero element
    probesOnRake = 1 + I[0][0]  # Number of probes on each rake
    thetas = thetasRaw[::probesOnRake]  # Creating new thetas array with no repeats

    pressures = data[:,3]
    radialAvg = []
    for i in range(len(thetas)):
        sum = 0
        for j in range(probesOnRake):
            sum = sum + pressures[i * probesOnRake + j]
        radialAvg = np.append(radialAvg, [sum / probesOnRake])

    return (radialAvg)
def circumferentialAverage(filename):
    """Returns circumferentially averaged total pressures from rake data. (One averaged value per span of probes)

    :param filename: The name of the .csv file containing pressure probe data. Column Format: [Probe Number, Span, Theta(radians), Pressure). Must be in same folder as areaWeightedAverage.py
    :type filename: string
    :return: Array of average pressures for each span value (in order of increasing span)
    :rtype: float array
    """

    if filename[-4:] != '.csv':
        filename = filename + '.csv'
    data = np.genfromtxt(filename, delimiter=',')
    data = np.delete(data, 0, axis=0)

    data = data[data[:, 1].argsort()]  # sorting by span
    data = data[data[:, 2].argsort(kind='mergesort')]  # sorting by theta

    thetasRaw = data[:, 2]  # Raw list of thetas from sorted csv (has repeats)
    diff_list = np.diff(thetasRaw)  # Finding differences between adjacent theta values
    I = np.nonzero(diff_list)  # Determining the location of first nonzero element
    probesOnRake = 1 + I[0][0]  # Number of probes on each rake
    thetas = thetasRaw[::probesOnRake]  # Creating new thetas array with no repeats

    pressures = data[:,3]
    circumferentialAvg = []
    for i in range(probesOnRake):
        sum = 0
        for j in range(len(thetas)):
            sum = sum + pressures[j * probesOnRake + i]
        circumferentialAvg = np.append(circumferentialAvg, [sum / probesOnRake])

    return (circumferentialAvg)


if __name__ == "__main__":
    print("Data column structure: ['Probe Number', 'Span', 'Theta (rad)', 'Pressure']")
    filename = input("Enter filename of dataset: ")

    whichType = input("Select average type (Radial = '1', Circumferential = '2', Weighted Area = '3': ")
    if whichType == '3':
        hub, casing = input("Enter hub radius followed by casing radius: ").split()
        print('Area weighted average pressure: ' + str(areaWeightedAvg(filename, float(hub), float(casing))))
    elif whichType == '1':
        print(radialAverage(filename))
    elif whichType == '2':
        print(circumferentialAverage(filename))
