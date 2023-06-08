import matplotlib.pyplot as plt
import numpy as np


my_data = np.genfromtxt('sample.csv', delimiter=',')
my_data = np.delete(my_data,0,axis=0)
print(my_data)
print(len(my_data))
radii = my_data[:,1]
thetas = my_data[:,2]
pressure = my_data[:,3]
print(thetas)


fig = plt.figure()
fig.set_size_inches(11, 8.5)
ax = fig.add_subplot(projection='polar')
c = ax.scatter(thetas,radii,s=60*(pressure-99))
testT = [1.70169605, 3.0106931,  4.31969035, 5.62868725, 6.8067842]
radi2 = [1, 1, 1, 1, 1]
d = ax.scatter(testT,radi2)
for i, value in enumerate(pressure):
    plt.annotate(str(round(value,1)), xy=(thetas[i],radii[i]), xytext=(.25, -10), textcoords='offset points', horizontalalignment='center', annotation_clip=False)

plt.show()