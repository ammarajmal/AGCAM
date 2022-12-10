'''
Sample Usage:-
python3 pose_gigecam.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''
marker_size = 0.053 # in meters
distance_from_marker = 0.88 # in meters

import cv2
import sys
from utils import ARUCO_DICT, aruco_dimensions
import argparse
import time
import mvsdk
import glob
import os
import numpy as np

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)
    # detected_markers = aruco_display(corners, ids, rejected, frame)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.053, matrix_coefficients,
                                                                       distortion_coefficients)
            cv2.putText(frame, f"X: {tvec[0][0][0]:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Y: {tvec[0][0][1]:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Z: {tvec[0][0][2]:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            rvectors = rvec[0][0]
            tvectors = tvec[0][0]

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 
            # Finding the dimensions of the markers
            aruco_dimensions(corners, ids, rejected_img_points, frame)
            # Draw Axis
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.03)
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.03)
            # cv2.putText(frame, f"X: {tvec[0][0][0]:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(frame, f"Y: {tvec[0][0][1]:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(frame, f"Z: {tvec[0][0][2]:.2f} m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    cal_image_count = 0
    if nDev < 1:
        print("No camera was found!")
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
    mvsdk.CameraSetExposureTime(hCamera, 7 * 1000)

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
    while True:
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        
            # Convert the data to OpenCV format
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if monoCamera else 3))
            frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_LINEAR)
            
            output = pose_esitmation(frame, aruco_dict_type, k, d)
            
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            
            cv2.putText(output, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Estimated Pose", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except mvsdk.CameraException as e:
            print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
            break
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
        

        
    
