'''
Sample Usage:-
python3 pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''
import sys
import math
import time
import argparse
import numpy as np
import cv2.aruco as aruco
import cv2
from utils import ARUCO_DICT, aruco_dimensions



# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

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
            rvec_list, tvec_list, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.053, matrix_coefficients,
                                                                       distortion_coefficients)
            # cv2.putText(frame, f"X: {tvec_list[0][0][0]:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(frame, f"Y: {tvec_list[0][0][1]:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Distance: {tvec_list[0][0][2]:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            rvec = rvec_list[0][0]
            tvec = tvec_list[0][0]
            print("rvec_list: ", rvec)
            print("tvec_list: ", tvec)
            
            aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.053)
            
            rvec_flipped = rvec * -1
            tvec_flipped = tvec * -1
            
            rotation_matrix, jacobian = cv2.Rodrigues(rvec_flipped)
            realworld_tvec = np.dot(rotation_matrix, tvec_flipped)
            
            pitch, roll, yaw = rotationMatrixToEulerAngles(rotation_matrix)
            
            tvec_str = f"X: {realworld_tvec[0]:.2f}, Y: {realworld_tvec[1]:.2f}, Z: {realworld_tvec[2]:.2f}"
            rvec_str = f"Roll: {math.degrees(roll):.0f}, Pitch: {round(math.degrees(pitch))}, Yaw: {math.degrees(yaw):.0f}"
            cv2.putText(frame, tvec_str, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 255), 2)
            cv2.putText(frame, rvec_str, (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 255), 2)
            
            
            
            

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
    pTime = time.time()
    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(output, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        cv2.imshow('Estimated Pose', output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()