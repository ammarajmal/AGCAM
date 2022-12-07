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

# def calibrate_gigecam(dirpath, square_size, width, height, visualize=False):
#     """gige camera calibration"""
#     # enumerate cameras
#     DevList = mvsdk.CameraEnumerateDevice()
#     nDev = len(DevList)
#     if nDev < 1:
#         print("No camera was found!")
#         return

#     for i, DevInfo in enumerate(DevList):
#         print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
#     i = 0 if nDev == 1 else int(input("Select camera: "))
#     DevInfo = DevList[i]
#     print(DevInfo)

# 	# Turn on the camera
#     hCamera = 0
#     try:
#         hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
#     except mvsdk.CameraException as excepttionnumber:
#         print("CameraInit Failed({}): {}".format(excepttionnumber.error_code, e.message) )
#         return

#     # Get camera feature description
#     cap = mvsdk.CameraGetCapability(hCamera)

#     # Determine whether it is a black and white camera or a color camera
#     monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

#     # Monochrome camera allows ISP to directly output MONO data
#     # instead of expanding to 24-bit grayscale with R=G=B
#     if monoCamera:
#         mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
#     else:
#         mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

#     # Camera mode switched to continuous acquisition
#     mvsdk.CameraSetTriggerMode(hCamera, 0)

#     # Manual exposure, exposure time 30ms
#     mvsdk.CameraSetAeState(hCamera, 0)
#     mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

#     # Let the SDK internal image fetching thread start to work
#     mvsdk.CameraPlay(hCamera)

#     # Calculate the required size of the RGB buffer,
#     # which is directly allocated according to the maximum resolution of the camera
#     FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

#     # Allocate RGB buffer to store the image output by ISP

#     # Remarks: RAW data is transmitted from the camera to the PC,
#     # which is converted to RGB data by the software ISP on the PC
#     # (if it is a black and white camera, there is no need to convert the format,
#     # but the ISP has other processing, so this buffer also needs to be allocated)
#     pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
#     pTime = time.time()
#     while (True):
#         # Take a frame from the camera
#         try:
#             pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
#             mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
#             mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

#             # Windows: The image data fetched under windows is upside down and stored in BMP format.
#             # To convert to opencv, you need to flip up and down to be positive

#             # Linux: Directly output positive under linux, no need to flip up and down
#             if platform.system() == "Windows":
#                 mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

#             # At this time, the picture has been stored in pFrameBuffer,
#             # for color camera pFrameBuffer=RGB data, for black and white camera pFrameBuffer=8-bit grayscale data
#             # Convert pFrameBuffer to opencv image format for subsequent algorithm processing
#             frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
#             frame = np.frombuffer(frame_data, dtype=np.uint8)
#             frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

#             frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
#             cTime = time.time()
#             fps = 1/(cTime-pTime)
#             pTime = cTime
#             cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
#             print(int(fps))
#             cv2.imshow("Press q to end", frame)

#         except mvsdk.CameraException as e:
#             if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
#                 print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

# 	# turn off the camera
#     mvsdk.CameraUnInit(hCamera)

# 	# free framebuffer
#     mvsdk.CameraAlignFree(pFrameBuffer)


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
    
    with open('GigEcameraParameters.txt', 'w') as f:
        f.write(mtx)
        f.write('\n')
        f.write(dist)
        
    print("rvecs: ", rvecs)
    print("tvecs: ", tvecs)

    # np.save("gigecam_calib.npy", [ret, mtx, dist, rvecs, tvecs])
    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)

    
