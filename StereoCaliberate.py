import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space

imgpoints_l = [] # 2d points in image plane.
imgpoints_r = [] # 2d points in image plane.

images_l = glob.glob('./data/left/*.jpg')

images_r = glob.glob('./data/right/*.jpg')


assert len(images_l) == len(images_r), 'There must be equal number of left and right photos.'

for i in range(len(images_l)):
	img_l = cv2.imread(images_l[i])
	gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
	ret_l, corners_l = cv2.findChessboardCorners(gray_l, (7,6),None)
	
	img_r = cv2.imread(images_r[i])
	gray_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
	ret_r, corners_r = cv2.findChessboardCorners(gray_r, (7,6),None)

	if ret_l and ret_r:
		objpoints.append(objp)

		cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),criteria2)
		cv2.cornerSubPix(gray_r,corners_r,(11,11),(-1,-1),criteria2)

		imgpoints_l.append(corners_l)
		imgpoints_r.append(corners_l)

imgsize = gray_r.shape[::-1]

ret_l, cameraMatrix_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, imgsize,None,None)
ret_r, cameraMatrix_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, imgsize,None,None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, \
	imgpoints_r, cameraMatrix_l, dist_l, cameraMatrix_r, dist_r, imgsize, cv2.CALIB_USE_INTRINSIC_GUESS, criteria = criteria2)

print ("Rotation Matrix:", R, "\n\n")
print ("Translation Vector:", T, "\n\n")
print ("Essential Matrix:", E, "\n\n")
print ("Fundamtel Matrix:", F, "\n\n")


R1 = R2 = np.empty([3,3])
P1 = P2 = np.empty([3,4])

Q = np.empty([4,4])

cv2.stereoRectify(cameraMatrix_l, dist_l, cameraMatrix_r, dist_r, imgsize, R, T, R1, R2, P1, P2, Q, 0, alpha = 1)

#print (R,'\n\n', T, '\n\n', R1,'\n\n', R2,'\n\n', P1,'\n\n', P2,'\n\n', Q)

#print (np.shape(map_l1), np.shape(map_l2))
#print (np.shape(map_r1), np.shape(map_r2))

print (imgsize)


for i in range(len(images_l)):
	img_l = cv2.imread(images_l[i])
	img_r = cv2.imread(images_r[i])
	
	h_l, w_l = img_l.shape[:2]
	h_r, w_r = img_r.shape[:2]

	newMatrix_l, roi_l=cv2.getOptimalNewCameraMatrix(cameraMatrix_l,dist_l,(w_l,h_l),1,(w_l,h_l))
	newMatrix_R, roi_r=cv2.getOptimalNewCameraMatrix(cameraMatrix_r,dist_r,(w_r,h_r),1,(w_r,h_r))

	map_l1, map_l2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, newMatrix_l, imgsize, cv2.CV_32FC1, None, None)
	map_r1, map_r2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R1, newMatrix_R, imgsize, cv2.CV_32FC1, None, None)

	#print (img_l)
	#cv2.imshow("img_l"+str(i),img_l)
	rec_l = cv2.remap(img_l, map_l1, map_l2, 1)
	x,y,w,h = roi_l
	rec_l = rec_l[y:y+h, x:x+w]
	cv2.imwrite('./results/lcalibresult'+str(i)+'.jpg',rec_l)

	rec_r = cv2.remap(img_r, map_r1, map_r2, 1)
	x,y,w,h = roi_r
	rec_r = rec_r[y:y+h, x:x+w]
	cv2.imwrite('./results/rcalibresult'+str(i)+'.jpg',rec_r)

stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(gray_l,gray_r)
plt.imshow(disparity,'gray')
plt.show()
