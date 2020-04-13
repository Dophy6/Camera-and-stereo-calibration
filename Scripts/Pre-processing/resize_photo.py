import os
from PIL import Image

CURRENT_PATH = os.path.abspath(os.getcwd())
PHOTO_DIRs = ["/Photo/Left", "/Photo/Right"]

for directory in PHOTO_DIRs:
	for filename in os.listdir(CURRENT_PATH + directory):
		if filename.endswith(".jpg"):
			temp = Image.open(CURRENT_PATH + directory + "/" + filename)
			
			width, height = temp.size # Get dimensions
			
			width /= 1
			height /= 1

			newsize = (int(width), int(height)) 
			temp = temp.resize(newsize) #Resize for better OCR read

			temp.save(CURRENT_PATH + directory + "/" + filename)