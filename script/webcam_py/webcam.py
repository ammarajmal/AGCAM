#!/usr/bin/env python

# Path: webcam.py
# Author: Ammar Ajmal
# Date: 2022-12-05
# Description: This script is used to capture images from a webcam and show them in a window.

import time
import cv2

cap = cv2.VideoCapture(2)
pTime = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
