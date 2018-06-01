import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


#im = cv2.imread("coba/daona2_2.jpg")

def canny(im):
	im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(im,20,255,L2gradient=False)  #edges = cv2.Canny(im,lower_threshold,upper_threshold,L2gradient=False)
	return edges


for file in glob.glob("coba" + "\\*.jpg"):
	print(file)
	im = cv2.imread(file)
	fv_canny = canny(im)

	#cv2.imshow('has',canny(im))
#cv2.imshow('hasil',canny(im))

#menghitung piksel putih
#n_white_pix = np.sum(canny(im) == 255)
	hit = cv2.countNonZero(canny(im))
	print('Number of white pixels:', hit)

cv2.waitKey(0)
cv2.destroyAllWindows()
