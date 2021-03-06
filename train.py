# import the necessary packages
import numpy as np
import os
import cv2
import glob
import csv
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import csv23


# fixed-sizes for image
fixed_size = tuple((300,300))

# path to training data
train_path = "./coba/"

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
# print(train_labels)

#=====================================================================

filename = open('cannypixel.csv', 'r')
dataframe = pandas.read_csv(filename)

kelas = dataframe.drop(dataframe.columns[:-1], axis=1)
data = dataframe.drop(dataframe.columns[-1:], axis=1)

print(data)
print(kelas)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

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

num = 102

# create all the machine learning models
models = []
models.append(('SVM', SVC(random_state=9)))

# variables to hold the results and names
results = []
names = []

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, data, kelas, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)



# create the model - Random Forests
clf  =  SVC(random_state=num)
# fit the training data to the model
clf.fit(data,kelas)

# path to test data
test_path = "./training/"

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_canny      = canny(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_canny, fv_hu_moments])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    cv2.putText(image,prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)


    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()




