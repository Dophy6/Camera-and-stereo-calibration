import logging
import numpy, cv2, os
from matplotlib import pyplot as plt

CURRENT_PATH = os.path.abspath(os.getcwd())
PHOTO_DIRs = ["/Photox100", "/Photox075", "/Photox050", "/Photox025", "/Photox010"]
COLOR_DIRs = ["/RGB", "/BW"]
CONTRAST_DIRs = ["/Contrastx100", "/Contrastx115", "/Contrastx130"]
CAMERA_DIRs = ["/Left", "/Right"]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(funcName)s]: %(message)s", datefmt="%d/%m %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("CAMERA CALIBRATION")

fh = logging.FileHandler("camera_cal.log", "a")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(funcName)s]: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

real_points = numpy.array([[numpy.float32(j),numpy.float32(i),numpy.float32(0)] for i in range(0, 14, 2) for j in range(0, 14, 2)])

for photo_dir in PHOTO_DIRs:
	width, height, channel = (0,0,0)
	for color_dir in COLOR_DIRs:
		for contrast_dir in CONTRAST_DIRs:
			for camera_dir in CAMERA_DIRs:
				p3d, p2d = [], []
				final_path = CURRENT_PATH + photo_dir + color_dir + contrast_dir + camera_dir
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
