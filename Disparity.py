import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
# 6*7 represents the size of chess board,you can change according to your own chessboard.
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_left = [] # 2d points in image plane.
imgpoints_right = [] # 2d points in image plane.

images1 = glob.glob('data/left/*.jpg')
images2 = glob.glob('data/right/*.jpg')

for i in range(12):
    # for fname in images:
    img_left = cv2.imread(images1[i])
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.imread(images2[i])
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    image_size = gray_left.shape[::-1]
    # Find the chess board corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (7, 6), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret_left and ret_right:
        objpoints.append(objp)

        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners2_left)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners2_right)

        # # Draw and display the corners
        # img_left = cv2.drawChessboardCorners(img_left, (7, 6), corners2_left, ret_left)
        # img_right = cv2.drawChessboardCorners(img_right, (7, 6), corners2_right, ret_right)
        # cv2.imshow('left img', img_left)
        # cv2.imshow('right img', img_right)
        # cv2.waitKey(0)

ret, mtx_left, dist_left, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None,None)
ret, mtx_right, dist_right, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None,None)


(rms_stereo, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = \
cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left, mtx_right, dist_right,  gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC);

# (width,height)
size = (640, 480)
# size of your picture
left = cv2.imread('data/left/left01.jpg',0)
right = cv2.imread('data/right/right01.jpg',0)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, size, R, T)
mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, size, cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, size, cv2.CV_16SC2)


left_img_remap = cv2.remap(left, mapL1, mapL2, cv2.INTER_LINEAR)
right_img_remap = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)

stereo = cv2.StereoSGBM_create(minDisparity = 0,
        numDisparities = 64, #num_disp,
        blockSize = 5, #16,
        P1 = 200,
        P2 = 400,
        disp12MaxDiff = 1, #1,
        uniquenessRatio = 0,
        speckleWindowSize = 300,
        speckleRange = 7
)

print('computing disparity...')
# disparity = stereo.compute(imgL, imgR)
disparity = stereo.compute(left_img_remap, right_img_remap).astype(np.float32) / 16.0

print ("saving disparity as disparity_image_sgbm.txt")
cv2.imwrite("./results/disparity.jpg", disparity)