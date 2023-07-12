import pandas as pd
from distorted import Distortion
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# csvDF = pd.read_csv('data/Ground vortex aerodynamics Figure10f.csv')
csvDF = pd.read_csv('data/figure14.csv')
distortionObj = Distortion(csvDF)


distortionObj.plot_quantity('Total Pressure')

# print(csvDF.getDataframe())
# print(distortionObj.getDF())

# print(distortionObj.get_circumferentialAverage())
# print(distortionObj.get_radialAverage())
# print(distortionObj.get_areaWeightedAverage())
# print(distortionObj.ARP1420PFAVEqualRingArea())
# print(distortionObj.ARP1420().to_string())
# print(distortionObj.RollsRoyceDC60())

# x, y = distortionObj.getRingData(1)


# plt.plot(x,y)
# plt.show()
# Graphing




# def ringsEqualArea(outerRadius, numberOfRings):
#     radiusMiddleCircle = outerRadius / np.sqrt(numberOfRings)
#
#     radiiList = [radiusMiddleCircle]
#     for i in range(numberOfRings-1):
#         radiiList = radiiList + [radiusMiddleCircle * np.sqrt(i + 2)]
#     return radiiList
# print(ringsEqualArea(1, 9))
