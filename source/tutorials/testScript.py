import pandas as pd
from distorted import Distortion
import matplotlib.pyplot as plt

csvDF = pd.read_csv('data/sample.csv')
distortionObj = Distortion(csvDF)


#print(csvDF.getDataframe())
#print(distortionObj.getDF())

#print(distortionObj.get_circumferentialAverage())
#print(distortionObj.get_radialAverage())
#print(distortionObj.get_areaWeightedAverage())
#print(distortionObj.ARP1420PFAVEqualRingArea())
#print(distortionObj.ARP1420().to_string())
distortionObj.RollsRoyceDC60()



#x, y = distortionObj.getRingData(1)


#plt.plot(x,y)
#plt.show()

#Graphing