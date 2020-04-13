import cv2, os, copy, logging
from matplotlib import pyplot as plt
import numpy as np

CURRENT_PATH = os.path.abspath(os.getcwd())

#PATH_WORST = "/Photox100/RGB/Contrastx100/Right"
PATH_BEST = "/Photox010/RGB/Contrastx115"
PATHs = [PATH_BEST + "/Right", PATH_BEST + "/Left"]

logging.basicConfig(
	format="%(asctime)s - %(levelname)s - [%(funcName)s]: %(message)s", datefmt="%d/%m %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("STEREO CALIBRATION")

real_points = np.array([[np.float32(j),np.float32(i),np.float32(0)] for i in range(0, 14, 2) for j in range(0, 14, 2)])

p2d_left = []
p2d_right = []
p3d = []

if len(os.listdir(CURRENT_PATH + PATHs[1])) == len(os.listdir(CURRENT_PATH + PATHs[0])):

	img_ids = [i in range(len(os.listdir(CURRENT_PATH + PATHs[1])))]

	img_left = [filename for filename in os.listdir(CURRENT_PATH + PATHs[1])]
	img_right = [filename for filename in os.listdir(CURRENT_PATH + PATHs[0])]

	valid_img_ids = []

	width_l, height_l, channel_l = (0,0,0)
	width_r, height_r, channel_r = (0,0,0)

	for i in img_ids:
		img_left=cv2.imread(CURRENT_PATH + PATHs[1] + "/" + img_left[i])
		img_right=cv2.imread(CURRENT_PATH + PATHs[0] + "/" + img_right[i])
		
		if width_l == 0 or height_l == 0 or channel_l == 0:
			width_l, height_l, channel_l = img_left.shape
		if width_r == 0 or height_r == 0 or channel_r == 0:
			width_r, height_r, channel_r = img_right.shape

		ret_left,left_corners=cv2.findChessboardCornersSB(img_left,(7,7))
		ret_right,right_corners=cv2.findChessboardCornersSB(img_right,(7,7))
		
		if ret_left and ret_right:
			valid_img_ids.append(i)
			p2d_right.append(right_corners)
			p2d_left.append(left_corners)
			p3d.append(real_points)

	ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(p3d, p2d_right, (height_r,width_r),None, None)
	ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(p3d, p2d_left, (height_l,width_l),None, None)

	retval, _, _, _, _, R, T, E, F=(
		cv2.stereoCalibrate(
			p3d,  
			p2d_left, 
			p2d_right, 
			mtx_l,
			dist_l,
			mtx_r,
			dist_r,
			(max(height_l,height_r),max(width_l,width_r)),
			flags=cv2.CALIB_FIX_INTRINSIC
		)
	)

	logger.info("STEREO CALIBRATION RESULT: {}".format(retval))

	R1, R2, P1, P2, Q, roi1, roi2 =(
		cv2.stereoRectify(
			mtx_l,
			dist_l, 
			mtx_r, 
			dist_r, 
			(max(height_l,height_r),max(width_l,width_r)), 
			R, 
			T
		)
	)

	img_left=cv2.imread(CURRENT_PATH + PATHs[1] + "/" +'IMG_20200405_203353.jpg')

	#newcameramtx_l, roi_l=cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(width_l,height_l), 0,(width_l,height_l))
	map1_x,map1_y=(
		cv2.initUndistortRectifyMap(
			mtx_l, 
			dist_l, 
			R1, 
			P1, 
			(max(height_l,height_r),max(width_l,width_r)), 
			cv2.CV_32FC1
		)
	)
	img_left_remapped=cv2.remap(
		img_left,
		map1_x,
		map1_y,
		cv2.INTER_CUBIC
	)
	x_l,y_l,width_l,height_l = roi1
	img_left_remapped = img_left_remapped[:max(width_l,width_r),:max(height_l,height_r)]

	img_right=cv2.imread(CURRENT_PATH + PATHs[0] + "/" +'20200405_203350.jpg')

	#newcameramtx_r, roi_r=cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(width_r,height_r), 0,(width_r,height_r))
	map2_x,map2_y=cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (max(height_l,height_r),max(width_l,width_r)), cv2.CV_32FC1)
	img_right_remapped=cv2.remap(img_right,map2_x,map2_y,cv2.INTER_CUBIC)
	x_r,y_r,width_r,height_r = roi2
	#img_right_remapped = img_right_remapped[:max(width_l,width_r),:max(height_l,height_r)]

	fig, axs = plt.subplots(2, 2, figsize=(8,6))
	fig.suptitle('Fabio Merola W82000188 - CV project - Stereo calibration: partial result\nSTEREO CALIBRATION RESULT: {} - Using cv2.findChessboardCorners(image,size)'.format(retval))

	axs[0, 0].set_title("Original left image")
	axs[0, 0].imshow(img_left[...,::-1])

	axs[0, 1].set_title("Undistorted left image")
	axs[0, 1].imshow(img_left_remapped[...,::-1])

	axs[1, 0].set_title("Original right image")
	axs[1, 0].imshow(img_right[...,::-1])

	axs[1, 1].set_title("Undistorted right image")
	axs[1, 1].imshow(img_right_remapped[...,::-1])

	for ax in axs.flat:
		ax.set(xlabel='Width [pixel]', ylabel='Height [pixel]')
		ax.label_outer()
		
	plt.savefig(CURRENT_PATH + "/" + "stereo_cal_plot_{}.jpeg".format("partialSB"))
	
	plt.show()

	out=np.hstack((img_left_remapped,img_right_remapped))
	plt.figure(figsize=(16,6))
	plt.imshow(out[...,::-1])
	plt.suptitle('Fabio Merola W82000188 - CV project - Stereo calibration: full result\nSTEREO CALIBRATION RESULT: {}\nUsing cv2.findChessboardCorners(image,size)'.format(retval))
	plt.xlabel('Width [pixel]')
	plt.ylabel('Height [pixel]')
	plt.savefig(CURRENT_PATH + "/" + "stereo_cal_plot_{}.jpeg".format("fullSB"))
	plt.show()

