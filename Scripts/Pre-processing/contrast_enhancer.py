import os
from PIL import Image, ImageEnhance

CURRENT_PATH = os.path.abspath(os.getcwd())
PHOTO_DIRs = ["/Photox100", "/Photox075", "/Photox050", "/Photox025", "/Photox010"]
COLOR_DIRs = ["/RGB", "/BW"]
CAMERA_DIRs = ["/Left", "/Right"]
for directory in PHOTO_DIRs:
	for color in COLOR_DIRs:
		for camera in CAMERA_DIRs:
			for filename in os.listdir(CURRENT_PATH + directory + color + "/Contrastx100" + camera):
				if filename.endswith(".jpg"):
					temp = Image.open(CURRENT_PATH + directory + color + "/Contrastx100" + camera + "/" + filename)
					enhancer = ImageEnhance.Contrast(temp)
					enhancer.enhance(1.30).save(CURRENT_PATH + directory + color + "/Contrastx130" + camera + "/" + filename)
					
