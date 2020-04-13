import os
from PIL import Image

CURRENT_PATH = os.path.abspath(os.getcwd())
PHOTO_DIRs = ["/Photox100", "/Photox075", "/Photox050", "/Photox025", "/Photox010", ]
COLOR_DIRs = ["/RGB"]
CAMERA_DIRs = ["/Left", "/Right"]

for directory in PHOTO_DIRs:
	for camera in CAMERA_DIRs:
		print("Start {}".format(CURRENT_PATH + directory + COLOR_DIRs[0] + camera))
		for filename in os.listdir(CURRENT_PATH + directory + COLOR_DIRs[0] + camera):
			if filename.endswith(".jpg"):
				temp = Image.open(CURRENT_PATH + directory + COLOR_DIRs[0] + camera + "/" + filename)
				temp = temp.convert('L')
				temp.save(CURRENT_PATH + directory + "/BW" + camera + "/" + filename)
