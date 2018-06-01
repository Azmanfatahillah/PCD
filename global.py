from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import glob
import csv
import pip

# fixed-sizes for image
fixed_size = tuple((300,300))

# path to training data
train_path = "./coba/"

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.30

# seed for reproducing same results
seed = 9


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

images_per_class = 300

#====================================================================

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def canny(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(image,20,255,L2gradient=False)  #edges = cv2.Canny(im,lower_threshold,upper_threshold,L2gradient=False)
	return edges


#def fd_haralick(image):
    # convert the image to grayscale
 #   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
 #   haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
  #  return haralick

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


#==================================================================

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    k = 1
    # loop over the images in each sub-folder
    # for x in range(1,images_per_class+1):
    # loop through the test images
    for file in glob.glob(dir + "/*.jpg"):
        # get the image file name
        # file = dir + "/" + str(x) + ".jpg"
        print(file)
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_canny = canny(image)
        fv_histogram  = fd_histogram(image)
        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_canny, fv_hu_moments,current_label])

        global_features.append(global_feature)

        i += 1
        k += 1
    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

print("[STATUS] completed Global Feature Extraction...")

#==========================================================================================

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# normalize the feature vector in the range (0-1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# rescaled_features = scaler.fit_transform(global_features)
# print "[STATUS] feature vector normalized..."

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))


with open('cannypixel.csv', 'wb') as myDaun:
    daun = csv.writer(myDaun, dialect='excel')
    daun.writerows(global_features)
myDaun.close()

print("[STATUS] end of training..")
