
"""
python3 readingCamPar.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy
"""

import argparse
import numpy as np

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    args = vars(ap.parse_args())



    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    
    k = np.array(k)


    print(k)
    print(d)
    with open('GigEcameraParameters.txt', 'w') as f:
        f.write(str(k))
        f.write('\n')
        f.write(str(d))