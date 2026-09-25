import numpy as np
import matplotlib.pyplot as plt
import os 

directory = os.path.dirname(os.path.realpath(__file__))

print("Loading file...")

my_data = np.genfromtxt(directory+"/MCMCSamples.csv", delimiter=',')

x_vals, y_vals = my_data.transpose()

print(len(x_vals))

# Not plotting all to remove freeze-in region
plt.hist2d(x_vals[1000::],y_vals[1000::],bins = 100)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()

fig,(ax1,ax2) = plt.subplots(nrows = 2)
#Checking freeze-in length
ax1.plot(x_vals[:2000])
ax2.plot(y_vals[:2000])
plt.show()


# Below was used for a 1D distribution but I currently want to plot a 2D distribution
"""
x = np.linspace(-10,10,1000)
y = 0.5*np.exp(-np.abs(x))

plt.plot(x,y)
plt.hist(my_data,bins = 1000,density=True)

plt.xlim(-10,10)
plt.show()
"""