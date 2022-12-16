import matplotlib.pyplot as plt
import numpy as np
# read data from a text file and plot it

def plot_data():
    # read data from a text file and plot it
    data = np.loadtxt('framerate.txt')
    plt.plot(data)
    plt.show()

plot_data()