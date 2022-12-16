#!/usr/bin/env python3

"""
    Name: gigecam_calib.py
    Description: function to calibrate the webcam and save the calibration parameters
    Author: Ammar Ajmal
    Date: 2022-12-07
    The calibration parameters are saved in the file 'gigecam_calib.npy'
    
    Sample Usage:-
    
    python3 gigecam_calib.py --dir calibration_checkerboard/ --square_size 0.024
    python3 gigecam_calib.py --dir calibration_checkerboard/ --square_size 0.024 --visualize True
"""

import time
import platform
import mvsdk
import glob
import os
import argparse
import cv2
import numpy as np


def calibrate(dirpath, square_size, width, height, visualize=False):
    """Apply camera calibration to a set of images"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 3D points in real world space
    objp = np.zeros((width * height, 3), np.float32)
    # 2D points in image plane
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    images = os.listdir(dirpath)

    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If ret is True, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # If visualize is True, draw and display the corners
            if visualize:
                img = cv2.drawChessboardCorners(
                    img, (width, height), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True,
                    help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("-w", "--width", type=int,
                    help="Width of checkerboard (default=9)",  default=9)
    ap.add_argument("-t", "--height", type=int,
                    help="Height of checkerboard (default=6)", default=6)
    ap.add_argument("-s", "--square_size", type=float,
                    default=1, help="Length of one edge (in metres)")
    ap.add_argument("-v", "--visualize", type=str, default="False",
                    help="To visualize each checkerboard image")
    args = vars(ap.parse_args())

    dirpath = args['dir']
    # 2.4 cm == 0.024 m
    # square_size = 0.024
    square_size = args['square_size']

    width = args['width']
    height = args['height']

    if args["visualize"].lower() == "true":
        visualize = True
    else:
        visualize = False

    ret, mtx, dist, rvecs, tvecs = calibrate(
        dirpath, square_size, width, height, visualize)

    # print("ret: ", ret)
    print("mtx: ", mtx)
    print("dist: ", dist)
    

        
    # print("rvecs: ", rvecs)
    # print("tvecs: ", tvecs)

    # np.save("gigecam_calib.npy", [ret, mtx, dist, rvecs, tvecs])
    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)
    
    # with open('GigEcameraParameters_new.txt', 'w') as f:
    #     f.write(str(mtx))
    #     f.write('\n')
    #     f.write(str(dist))

    
