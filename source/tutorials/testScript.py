import pandas as pd
from distorted import Distortion
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

csvDF = pd.read_csv('data/.csv')
#csvDF = pd.read_csv('data/figure14.csv')
#csvDF2 = pd.read_csv('data/figure15NonDimensional.csv')
#csvDF = pd.read_csv('data/test2.csv')
distortionObj = Distortion(csvDF)
#distortionObj2 = Distortion(csvDF2)

distortionObj.plot_quantity('Total Pressure')
#distortionObj2.plot_quantity('Total Pressure')

# print(csvDF.getDataframe())
# print(distortionObj.getDF())


#print(distortionObj.ARP1420().to_string())
#print(distortionObj.RollsRoyceDC60())
#print(distortionObj.RollsRoyceDeltaPDeltaPAvg())
print(distortionObj.PrattAndWhitneyKD2())


# def ringsEqualArea(outerRadius, numberOfRings):
#     radiusMiddleCircle = outerRadius / np.sqrt(numberOfRings)
#
#     radiiList = [radiusMiddleCircle]
#     for i in range(numberOfRings-1):
#         radiiList = radiiList + [radiusMiddleCircle * np.sqrt(i + 2)]
#     return radiiList
# print(ringsEqualArea(1, 5))
