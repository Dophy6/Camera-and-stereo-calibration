import numpy, cv2, os, copy, logging
from matplotlib import pyplot as plt

CURRENT_PATH = os.path.abspath(os.getcwd())

PATH_WORST = "/Photox100/RGB/Contrastx100/Right"
PATH_BEST = "/Photox010/RGB/Contrastx115/Right"
PATHs = [PATH_BEST, PATH_WORST]

logging.basicConfig(
	format="%(asctime)s - %(levelname)s - [%(funcName)s]: %(message)s", datefmt="%d/%m %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("CAMERA CALIBRATION")

real_points = numpy.array([[numpy.float32(j),numpy.float32(i),numpy.float32(0)] for i in range(0, 14, 2) for j in range(0, 14, 2)])

for path in PATHs:
	width, height, channel = (0,0,0)	
	p3d, p2d = [], []
	final_path = CURRENT_PATH + path 
	for filename in os.listdir(final_path):
		if filename.endswith(".jpg"):
			img = cv2.imread(final_path + '/' + filename)
			if width == 0 or height == 0 or channel == 0:
				width, height, channel = img.shape
			ret, corners = cv2.findChessboardCorners(img, (7,7))
			if ret:
				logger.info("{} - Chessboard found in file : {}".format(final_path, filename))
				p3d.append(real_points)
				p2d.append(corners)
			else:
				logger.info("{} - Chessboard not found in file : {}".format(final_path, filename))
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(p3d, p2d, (height,width),None, None)
	logger.info("CALIBRATION RESULT: {}, RET: {}".format(final_path, ret))
	
	img=cv2.imread(final_path + '/' +'20200405_203350.jpg')[...,::-1]
	ret, corners = cv2.findChessboardCorners(img, (7,7))
	img_draw = img.copy()
	cv2.drawChessboardCorners(img_draw, (7,7), corners, ret)
	img_undistorted=cv2.undistort(img, mtx, dist)
	ret_u, corners_u = cv2.findChessboardCorners(img_undistorted, (7,7))
	img_undistorted_draw = img_undistorted.copy()
	cv2.drawChessboardCorners(img_undistorted_draw, (7,7), corners_u, ret_u)

	fig, axs = plt.subplots(2, 2, figsize=(8,6))
	fig.suptitle('Fabio Merola W82000188 - CV project - Camera calibration: {}'.format("best result" if path == PATH_BEST else "worst result"))

	axs[0, 0].set_title("Original image")
	axs[0, 0].imshow(img)

	axs[0, 1].set_title("Original image drawn")
	axs[0, 1].imshow(img_draw)

	axs[1, 0].set_title("Undistorted image")
	axs[1, 0].imshow(img_undistorted)

	axs[1, 1].set_title("Undistorted image drawn")
	axs[1, 1].imshow(img_undistorted_draw)

	for ax in axs.flat:
		ax.set(xlabel='Width [pixel]', ylabel='Height [pixel]')
		ax.label_outer()
		
	plt.savefig(CURRENT_PATH + "/" + "camera_cal_plot_{}.jpeg".format("best" if path == PATH_BEST else "worst"))
	
	plt.show()