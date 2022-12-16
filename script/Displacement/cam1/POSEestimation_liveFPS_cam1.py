import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
import imutils
import numpy as np
import matplotlib.animation as animation

import time
import cv2

def cameraRealTimeFPSPlot(filename):
    plt.style.use('fivethirtyeight')
    x_vals = []
    y_vals = []
    index = count()
    # read_fps = fps
    # read fps values from fps_webcam.txt
    # with open('fps_webcam.txt', 'r') as f:
    #     for line in f:
    #         read_fps.append(float(line.strip()))

    def animate(i):
        x_vals.append(next(index))
        with open(filename, 'r') as f:
            for line in f:
                y_vals.append(float(line.strip()))
                # y_vals.append(fps)
                plt.cla()
                plt.plot(x_vals, y_vals)

    ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()


cap = cv2.VideoCapture(0)
pTime = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('frame', frame)
    print(fps)
    with open('fps_webcam.txt', 'a') as f:
        f.write(str(int(fps))+'\n')
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cameraRealTimeFPSPlot(filename='fps_webcam.txt')
cap.release()
cv2.destroyAllWindows()


    
# cameraRealTimeFPSPlot()
    