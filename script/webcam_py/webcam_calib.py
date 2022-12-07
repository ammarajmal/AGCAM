#!/usr/bin/env python3

# Path: webcam_calib.py
#  Author: Ammar Ajmal
#  Date: 2022-12-05
#  Description: This script is used to calibrate the webcam and save the calibration parameters.

#  The calibration parameters are saved in the file "camera_calib.npy".
import time
import glob
import os
import cv2
import numpy as np



def cam_calibraiton():
    # "cam_calibraiton"  -  function to calibrate the webcam and save the calibration parameters

    frame_per_second = 30
    cv2.namedWindow("Camera Calibration")
    cv2.moveWindow("Camera Calibration", 800, 0)
    video_capture = cv2.VideoCapture(2)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, frame_per_second)

    prev_frame_time = time.time()
    cal_image_count = 0
    copyFrame = None
    print('Press "Space" to save an image, "Enter" to start camera calibration, and  "Esc" or "q" to quit')

    while (True):
        ret, frame = video_capture.read()
        copyFrame = frame.copy()

        # 1. Save an image(Space Key) -- if we see a valid checkerboard image
        # 2. Start Camera clibration (Enter Key) -- if we wanted to start camera calibration so long as we have enough images like 10 or 15
        # 3. Exit(Escape Key)

        inputKey = cv2.waitKey(frame_per_second)
        
        # find chessboard corners and draw then on the frame 
        ret, corners = cv2.findChessboardCorners(frame, (9,6), None)
        if ret == True:
            cv2.drawChessboardCorners(frame, (9,6), corners, ret)
        cv2.imshow("Camera Calibration", frame)
        

        if inputKey == ord(' '):
            print('-- Space pressed... saving image #'+str(cal_image_count)+'.jpg')
            path = os.getcwd()+"/images/"
            cv2.imwrite(path+"cal_image_"+str(cal_image_count)+".jpg", copyFrame)
            cal_image_count += 1

            

            
        elif inputKey == 13:
            print('-- Enter pressed... Starting Camera Calibaration')
            cb_width = 9
            cb_height = 6
            cb_square_size = 24.4

            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points
            cb_3D_points = np.zeros((cb_width * cb_height, 3), np.float32)
            cb_3D_points[:, :2] = np.mgrid[0:cb_width, 0:cb_height].T.reshape(-1, 2) * cb_square_size

            # Arrays to store object points and image points from all the images
            list_cb_3d_points = []  # 3d points in real world
            list_cb_2d_img_points = []  # 2d points in image plane
            
            # os.getcwd()
            path = os.getcwd()+"/images/"
            for frame_name in glob.glob(os.path.join(path, '*.jpg')):
                img = cv2.imread(frame_name)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (cb_width, cb_height), None)
                # if found, add object points, image points (after refining them)
                if ret is True:
                    list_cb_3d_points.append(cb_3D_points)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    list_cb_2d_img_points.append(corners2)
                    # Draw adn display the corners
                    shape_image = gray.shape[::-1]
                    cv2.drawChessboardCorners(img, (cb_width, cb_height), corners2, ret)
                    cv2.imshow('Camera Calibration', img)
                    cv2.waitKey(200)
            cv2.destroyAllWindows()
            print('Calibrating Camera... Please wait...')

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(list_cb_3d_points, list_cb_2d_img_points, (640, 480), None, None)
            
            print("Camera Calibration Done! Saving the calibration parameters to file..\n")
            with open('camera_calib.npy', 'wb') as f:
                np.save(f, mtx)
                np.save(f, dist)

            print('Camera Calibration Parameters Saved!\n')
            print('Calibration Matrix: ')
            print(mtx)
            print('Distortion: ')
            print(dist)
            print(' Camera Calibration Completed, file saved as "camera_calib.npy"!')
            break

        elif inputKey == ord('q') or inputKey == 27:
            print('-- Quitting...')
            break
    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    cam_calibraiton()