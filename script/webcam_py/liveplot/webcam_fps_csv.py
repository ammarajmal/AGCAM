#!/usr/bin/env python

# Path: webcam_fps_csv.py
# Author: Ammar Ajmal
# Date: 2022-12-17
# Description: This script calculates the FPS of the webcam and saves it in a csv file
import csv
import time
import cv2

field_names = ['ind', 'fpsCam1']
with open('fps_webcam.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
    csv_writer.writeheader()
cap = cv2.VideoCapture(0)
pTime = 0
ind = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    cv2.putText(frame, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('frame', frame)
    with open('fps_webcam.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        info = {
            'ind': ind,
            'fpsCam1': fps
        }
        csv_writer.writerow(info)
        print(ind, fps)
    pTime = cTime
    ind += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()