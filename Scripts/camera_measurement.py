import numpy, cv2, os, copy, logging, math
from matplotlib import pyplot as plt

CURRENT_PATH = os.path.abspath(os.getcwd())

#PATH_WORST = "/Photox100/RGB/Contrastx100/Right"
PATH_BEST = "/Left"
#PATHs = [PATH_BEST, PATH_WORST]

logging.basicConfig(
	format="%(asctime)s - %(levelname)s - [%(funcName)s]: %(message)s", datefmt="%d/%m %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("CAMERA CALIBRATION")

real_points = numpy.array([[numpy.float32(j),numpy.float32(i),numpy.float32(0)] for i in range(0, 14, 2) for j in range(0, 14, 2)])

width, height, channel = (0,0,0)	
p3d, p2d = [], []
final_path = CURRENT_PATH + PATH_BEST 
for filename in os.listdir(final_path):
	if filename.endswith(".jpg"):
		img = cv2.imread(final_path + '/' + filename)
		if width == 0 or height == 0 or channel == 0:
			width, height, channel = img.shape
		ret, corners = cv2.findChessboardCornersSB(img, (7,7))
		if ret:
			logger.info("{} - Chessboard found in file : {}".format(final_path, filename))
			p3d.append(real_points)
			p2d.append(corners)
		else:
			logger.info("{} - Chessboard not found in file : {}".format(final_path, filename))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(p3d, p2d, (height,width),None, None)
logger.info("CALIBRATION RESULT: {}, RET: {}".format(final_path, ret))

img=cv2.imread(final_path + '/' +'IMG_20200405_203414.jpg')[...,::-1]

img_undistorted=cv2.undistort(img, mtx, dist)
ret_u, corners_u = cv2.findChessboardCorners(img_undistorted, (7,7))

rmatrix, jacobian = cv2.Rodrigues(rvecs[3])

Rt = numpy.column_stack((rmatrix,tvecs[3]))

P_mtx = mtx.dot(Rt) #Projective matrix

s_sum = 0

for i in range(len(real_points)):
	XYZ1=numpy.array([[real_points[i,0],real_points[i,1],real_points[i,2],1]], dtype=numpy.float32)
	XYZ1=XYZ1.T #Transpose
	suv1=P_mtx.dot(XYZ1) #Matrix multiplication
	s=suv1[2,0]
	s_sum += s    

s_mean = s_sum / len(corners_u)

inv_mtx = numpy.linalg.inv(mtx)
inv_r_matrix = numpy.linalg.inv(rmatrix)

p_idxs = [0,5,12,31,8] #Point indexes 
calc_points = []
for p in p_idxs:
	'''
	print(corners_u[p])
	print(corners_u[p,0,0])
	print(corners_u[p,0,1])
	'''
	uv_1 = numpy.array([[corners_u[p,0,0],corners_u[p,0,1],1]], dtype=numpy.float32)

	uv_1 = uv_1.T

	suv_1 = s_mean * uv_1

	xyz_c = inv_mtx.dot(suv_1)
	
	xyz_c=xyz_c-tvecs[3]	
	
	XYZ=inv_r_matrix.dot(xyz_c)

	calc_points.append(XYZ)
	#print(XYZ)

p_0 = {"x":calc_points[0][0,0],"y":calc_points[0][1,0]}
p_5 = {"x":calc_points[1][0,0],"y":calc_points[1][1,0]}
p_12 = {"x":calc_points[2][0,0],"y":calc_points[2][1,0]}
p_31 = {"x":calc_points[3][0,0],"y":calc_points[3][1,0]}
p_8 = {"x":calc_points[4][0,0],"y":calc_points[4][1,0]}



dist1 = math.sqrt( ( ( p_0["x"]-p_5["x"] )**2 ) + ( ( p_0["y"]-p_5["y"] )**2 ) )

dist2 = math.sqrt( ( ( p_0["x"]-p_12["x"] )**2 ) + ( ( p_0["y"]-p_12["y"] )**2 ) )

dist3 = math.sqrt( ( ( p_31["x"]-p_8["x"] )**2 ) + ( ( p_31["y"]-p_8["y"] )**2 ) )

img_undistorted_draw_1 = img_undistorted.copy()
cv2.drawChessboardCorners(img_undistorted_draw_1, (1,1), corners_u[0], ret_u)
cv2.drawChessboardCorners(img_undistorted_draw_1, (1,1), corners_u[5], ret_u)
img_undistorted_draw_1 = cv2.line(img_undistorted_draw_1, (corners_u[0,0,0],corners_u[0,0,1]), (corners_u[5,0,0],corners_u[5,0,1]), (255, 0, 0),3)

img_undistorted_draw_2 = img_undistorted.copy()
cv2.drawChessboardCorners(img_undistorted_draw_2, (1,1), corners_u[0], ret_u)
cv2.drawChessboardCorners(img_undistorted_draw_2, (1,1), corners_u[12], ret_u)
img_undistorted_draw_2 = cv2.line(img_undistorted_draw_2, (corners_u[0,0,0],corners_u[0,0,1]), (corners_u[12,0,0],corners_u[12,0,1]), (255, 0, 0),3)

img_undistorted_draw_3 = img_undistorted.copy()
cv2.drawChessboardCorners(img_undistorted_draw_3, (1,1), corners_u[31], ret_u)
cv2.drawChessboardCorners(img_undistorted_draw_3, (1,1), corners_u[8], ret_u)
img_undistorted_draw_3 = cv2.line(img_undistorted_draw_3, (corners_u[31,0,0],corners_u[31,0,1]), (corners_u[8,0,0],corners_u[8,0,1]), (255, 0, 0),3)


fig, axs = plt.subplots(2, 3, figsize=(12,9))
fig.suptitle('Fabio Merola W82000188 - CV project - Measurement test\nRe-projection err.: {} - Mean scale factor: {}'.format(ret,s_mean))

axs[0,0].set_title("1. Measured dist.: {}".format(dist1))
axs[0,0].imshow(img_undistorted_draw_1)

axs[0,1].set_title("2. Measured dist.: {}".format(dist2))
axs[0,1].imshow(img_undistorted_draw_2)

axs[0,2].set_title("3. Measured dist.: {}".format(dist3))
axs[0,2].imshow(img_undistorted_draw_3)

axs[1,0].set_title("1. Zoom-in")
axs[1,0].imshow(img_undistorted_draw_1)

axs[1,1].set_title("2. Zoom-in")
axs[1,1].imshow(img_undistorted_draw_2)

axs[1,2].set_title("3. Zoom-in")
axs[1,2].imshow(img_undistorted_draw_3)

for ax in axs.flat:
	ax.set(xlabel='Width [pixel]', ylabel='Height [pixel]')
	ax.label_outer()
	
plt.show()