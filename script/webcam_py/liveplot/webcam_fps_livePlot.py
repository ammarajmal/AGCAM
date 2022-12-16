#!/usr/bin/env python

# Path: webcam_fps_livePlot.py
# Author: Ammar Ajmal
# Date: 2022-12-17
# Description: This script plots the FPS of the webcam in real time 

import random
from itertools import count 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('fivethirtyeight')


def animate(i):
    data = pd.read_csv('fps_webcam.csv')
    x = data['ind']
    y1 = data['fpsCam1']
    
    plt.cla()
    plt.plot(x, y1, label='FPS')

ani = FuncAnimation(plt.gcf(), animate, interval=100)

plt.tight_layout()
plt.show()