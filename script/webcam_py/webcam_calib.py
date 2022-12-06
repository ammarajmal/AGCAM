#!/usr/bin/env python

# Path: webcam_calib.py
#  Author: Ammar Ajmal
#  Date: 2022-12-05
#  Description: This script is used to calibrate the webcam and save the calibration parameters.
#  The calibration parameters are used to undistort the images captured from the webcam.
#  The calibration parameters are saved in a file called "calib_params.npz".
#  The calibration parameters are used in the script "webcam.py".
#  The calibration parameters are used in the script "webcam_undistort.py".
#  The calibration parameters are used in the script "webcam_undistort_save.py".
#

import numpy as np
import cv2
import numpy as np
import time
import glob
ret, mtx, dist, rvecs, tvecs = [], [], [], [], []
# "cam_calibraiton"  -  function to calibrate the webcam and save the calibration parameters

def cam_calibraiton():
    ret, mtx, dist, rvecs, tvecs = [], [], [], [], []
    frame_per_second = 30
    cv2.namedWindow("Camera Calibration")
    cv2.moveWindow("Camera Calibration", 800, 0)
    video_capture = cv2.VideoCapture(2)

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, frame_per_second)

    prev_frame_time = time.time()
    cal_image_count = 0
    frame_count = 0
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
            cv2.imwrite("cal_image_"+str(cal_image_count)+".jpg", copyFrame)
            cal_image_count += 1
            new_frame_time = time.time()
            fps = 1/(new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame,
                        "FPS:" + str(int(fps)),
                        (10, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (100, 255, 0),
                        2,
                        cv2.LINE_AA)
            
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

            list_images = glob.glob('*.jpg')
            for frame_name in list_images:
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
                    cv2.imshow('img', img)
                    cv2.waitKey(500)
            cv2.destroyAllWindows()

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
            print(' Task Completed!')
            

        
        # elif inputKey == ord('u'):
        #     print('-- U pressed... Undistorting image...')
        #     img = cv2.imread('cal_image_0.jpg')
        #     h, w = img.shape[:2]
        #     newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            
        #     # undistort
        #     dst = cv2.undistort(img, mtx, dist, None, newCameraMatrix)
        #     # crop the image 
        #     x, y , w, h = roi
        #     dst = dst[y:y+h, x:x+w]
        #     cv2.imwrite('cal_image_0_undistort.png', dst)
            
            

        elif inputKey == ord('q') or inputKey == 27:
            print('-- Quitting...')
            break
    video_capture.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    cam_calibraiton()
