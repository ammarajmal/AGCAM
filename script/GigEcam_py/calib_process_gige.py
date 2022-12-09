#!/usr/bin/env python3

#  Path: calib_process_gige.py
#  Author: Ammar Ajmal
#  Date: 2022-12-08
#  Description: This script is used to calibrate the gige and save the calibration parameters.

#  The calibration parameters are saved in the file "gige_camera_calib.npy".
import time
import glob
import os
import cv2
import numpy as np
import mvsdk
import platform

def cam_calibration():
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    cal_image_count = 0
    if nDev < 1:
        print("No camera was found!")
        return
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]
    print(DevInfo)
    
    # Turn on the camera
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as myexcept:
        print("CameraInit Failed({}): {}".format(myexcept.error_code, myexcept.message) )
        return
    # Get camera feature description
    cap = mvsdk.CameraGetCapability(hCamera)
    
    # Determine whether it is a black and white camera or a color camera
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # Monochrome camera allows ISP to directly output MONO data 
    # instead of expanding to 24-bit grayscale with R=G=B
    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    # Camera mode switched to continuous acquisition
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # Manual exposure, exposure time 30ms
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 10 * 1000)

    # Let the SDK internal image fetching thread start to work
    mvsdk.CameraPlay(hCamera)

    # Calculate the required size of the RGB buffer, 
    # which is directly allocated according to the maximum resolution of the camera
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

	# Allocate RGB buffer to store the image output by ISP
	
 	# Remarks: RAW data is transmitted from the camera to the PC, 
 	# which is converted to RGB data by the software ISP on the PC 
	# (if it is a black and white camera, there is no need to convert the format,
	# but the ISP has other processing, so this buffer also needs to be allocated)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    pTime = time.time()
    option = cv2.waitKey(1) & 0xFF
    while (option != ord('q')):
        # Get one frame from the camera
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        
            # Convert the data to OpenCV format
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if monoCamera else 3))
            frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_LINEAR)
            
            # if not monoCamera:
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            copyFrame = frame.copy()
            ret, corners = cv2.findChessboardCorners(frame, (9,6), None)
            if ret == True:
                cv2.drawChessboardCorners(frame, (9,6), corners, ret)
            cv2.imshow("Camera Calibration", frame)
            
            if option == ord(' '):
                print('-- Space pressed... saving image #'+str(cal_image_count)+'.jpg')
                path = os.getcwd()+"/calibration_checkerboard/"
                cv2.imwrite(path+"cal_image_"+str(cal_image_count)+".jpg", copyFrame)
                cal_image_count += 1
            
            
            
            
            
            option = cv2.waitKey(1) & 0xFF
        
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))    
    
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)

def main():
    try:
        cam_calibration()
    finally:
        cv2.destroyAllWindows()
        print("Camera Calibration Completed")

main()

    
        
    
    
    
    # frame_per_second = 30
    # cv2.namedWindow("Camera Calibration")
    # cv2.moveWindow("Camera Calibration", 800, 0)
    