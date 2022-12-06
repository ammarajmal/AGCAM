#!/usr/bin/env python

# Path: webcam_calib.py
#  Author: Ammar Ajmal
#  Date: 2022-12-06
#  Description: This script is used to show the pose of the chessboard using camera parameters.
#  The calibration parameters are read  from the file 'camera_calib.npy'
#  The chessboard pose is shown in the image window 'Chessboard Pose'

import numpy as np
import cv2
import numpy as np
import time
import glob
ret, mtx, dist, rvecs, tvecs = [], [], [], [], []
# "cam_calibraiton"  -  function to calibrate the webcam and save the calibration parameters

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


def pose_():
    path = '/home/ammar/sitl/AGCAM/images/'
    with open(path+'camera_calib.npy', 'rb') as f:
        camera_matrix = np.load(f)
        camera_distortion = np.load(f)
    
    chessboard_size = (9, 6)
    framesize = (640, 480)

    # termination criteria 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)  # 3D points in real world space   # 9x6 = 54   54x3 = 162  162x1 = 162   
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)  # 2D points in image plane        # 9x6 = 54   54x2 = 108  108x1 = 108
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    
    video_capture = cv2.VideoCapture(2)         
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
             corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
             
            #  Find the rotation and translation vectors.
             ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, camera_distortion)
             
            #  project 3D points to image plane
             imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, camera_distortion)
             
             img = draw(frame, corners2, imgpts)
        cv2.imshow("Chessboard Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == '__main__':
    pose_()