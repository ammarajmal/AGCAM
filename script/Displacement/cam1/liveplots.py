import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')


index = count()

def animate(i):
    data = pd.read_csv('fps.csv')
    # info = {"ind": index_value, "fps": fps_value}
    x = data['ind']
    y = data['fps']


    print(y)
    plt.cla()
    plt.plot(x, y, label='Realtime FPS')
    plt.legend(loc='upper left')
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval=200)

plt.tight_layout()
plt.show()

