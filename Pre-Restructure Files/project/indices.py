"""
indices.py
==========
Inlet Distortion index functions
"""

import numpy as np
import averagingMethods as am

def pDeltaPavg1(data):
    """Returns a simple distortion index

    :math:`\\frac{\\Delta P_{max-min}}{\\bar{P}}=\\frac{P_{max}-P_{min}}{P_{avg}}`

    :math:`P_{max}` = Maximum inlet total pressure

    :math:`P_{min}` = Minimum inlet total pressure

    :math:`P_{avg}` = Average inlet total pressure


    :param data: Total Pressure rake data
    :type data: Pandas dataframe
    :return: Calculated Distortion Index
    :rtype: float
    """

    pressures = data[:, 3]
    pMin = np.min(pressures)
    pMax = np.max(pressures)
    pAvg = np.average(pressures)

    index = (pMax - pMin) / pAvg

    return index


def pDeltaPavg2(data):
    """Returns a simple distortion index

    :math:`\\frac{\\Delta P_{avg-min}}{\\bar{P}}=\\frac{P_{avg}-P_{min}}{P_{avg}}`

    :math:`P_{min}` = Minimum inlet total pressure

    :math:`P_{avg}` = Average inlet total pressure


    :param data: Total Pressure rake data
    :type data: Pandas dataframe
    :return: Calculated Distortion Index
    :rtype: float
    """

    pressures = data[:, 3]
    pMin = np.min(pressures)
    pAvg = np.average(pressures)

    index = (pAvg - pMin) / pAvg

    return index


def RRCriticalAngle(data):
    """Returns the Rolls Royce critical angle index
    Work in Progress


    :param data: Total Pressure rake data
    :type data: Pandas dataframe
    :return: Calculated Distortion Index
    :rtype: float
    """
