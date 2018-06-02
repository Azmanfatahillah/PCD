import cv2
import glob
import numpy as np

#images = [cv2.imread(file) for file in glob.glob("path/to/files/*.jpg")]
images = cv2.imread('hevea brasilinsis_50.jpg')
small = cv2.resize(images, (0, 0), fx=0.2, fy=0.2)

def fd_morph(images):
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening

hit = cv2.countNonZero(fd_morph(images))
print('Number of white pixels:', hit)

cv2.imshow('hasil', fd_morph(images))
cv2.waitKey(0)
cv2.destroyAllWindows()
