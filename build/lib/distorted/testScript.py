from csvToDataframe import CSVToDataframe
from distortion import Distortion
import matplotlib.pyplot as plt

csvDF = CSVToDataframe('figure15')
distortionObj = Distortion(csvDF.getDataframe())

distortionObj.hubRadius = 1
distortionObj.casingRadius = 5

#print(csvDF.getDataframe())
#print(distortionObj.getDF())

#print(distortionObj.get_circumferentialAverage())
#print(distortionObj.get_radialAverage())
#print(distortionObj.get_areaWeightedAverage())
#print(distortionObj.ARP1420PFAVEqualRingArea())
print(distortionObj.ARP1420().to_string())



#x, y = distortionObj.getRingData(1)


#plt.plot(x,y)
#plt.show()

#Graphing
