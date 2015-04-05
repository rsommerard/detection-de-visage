# coding=utf-8

import cv2
import sys
import os
import random
import math
import numpy

###########################
# Création des eigenfaces #
###########################

CASC_PATH = 'haarcascade_frontalface_default.xml'
IMAGES_PATH = 'faces/'
EIGENFACES_PATH = 'eigenfaces/'

faceCascade = cv2.CascadeClassifier(CASC_PATH)

for dirname in os.listdir(IMAGES_PATH):
    if dirname == '.DS_Store':
        continue

    i = 0
    for filename in os.listdir(IMAGES_PATH + dirname):
        if filename == '.DS_Store':
            continue

        image = cv2.imread(IMAGES_PATH + dirname + '/' + filename)
        print(IMAGES_PATH + dirname + '/' + filename)

        faces = faceCascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5,
            minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

        print "Found {0} faces!".format(len(faces))

        if not os.path.exists(EIGENFACES_PATH + dirname):
            os.makedirs(EIGENFACES_PATH + dirname)


        for (x, y, w, h) in faces:
            resized_image = cv2.resize(image[y:(y+h), x:(x+w)], (100, 100))
            cv2.imwrite(EIGENFACES_PATH + dirname + '/' + str(i) + '.jpeg', resized_image)
            i += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

print('-' * 80)

######################################
# Création du csv pour le Classifier #
######################################

CSV_FILE = 'faces.csv'
SEPARATOR = ";"
FACES_FOLDER = "faces/"

label = 0
content = ''

for dirname, dirnames, filenames in os.walk(FACES_FOLDER):
    if dirname == '.DS_Store':
        continue

    for subdirname in dirnames:
        if subdirname == '.DS_Store':
            continue

        subject_path = os.path.join(dirname, subdirname)
        for filename in os.listdir(subject_path):
            if filename == '.DS_Store':
                continue
            abs_path = "%s/%s" % (subject_path, filename)
            content += "%s%s%s\n" % (abs_path, SEPARATOR, label)

        label += 1

with open(CSV_FILE, 'w') as file:
    file.write(content)

########################################
# Apprentissage et prédiction du sujet #
########################################

with open(CSV_FILE, 'r') as file:
    csv = file.readlines()

ratio = 0.1
random.shuffle(csv)
size = int(math.floor(ratio*len(csv)))
training_data, testing_data = csv[size:], csv[:size]

print('Nb. training data: ' + str(len(training_data)))
print('Nb. testing data: ' + str(len(testing_data)))

print('-' * 40)

data = {}

for line in training_data:
    filename, label = line.strip().split(';')

    if int(label) in data:
        numpy.append(data[int(label)], cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    else:
        data[int(label)] = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)

model = cv2.createEigenFaceRecognizer()
model.train(data.values(), numpy.array(data.keys()))

for line in testing_data:
    filename, label =  line.strip().split(';')
    predicted_label = model.predict(cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    print 'Predicted: %(predicted)s  Actual: %(actual)s' %  {"predicted": predicted_label[0], "actual": label}
