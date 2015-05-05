# coding=utf-8

import cv2
import os
import random
import math
import numpy

###########################
# Création des eigenfaces #
###########################

print('-' * 80)
print('# Détection des visage puis création des eigenfaces pour permettre la reconnaissance de visage.')
print('-' * 80)

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

        # Chargement de l'image
        image = cv2.imread(IMAGES_PATH + dirname + '/' + filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        print('> ' + IMAGES_PATH + dirname + '/' + filename)

        faces = faceCascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5,
            minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

        print ">> Trouvé {0} visage(s) !".format(len(faces))
        print ">>> Eigenface saved in " + EIGENFACES_PATH + dirname

        if(len(faces) > 1):
            continue

        if not os.path.exists(EIGENFACES_PATH + dirname):
            os.makedirs(EIGENFACES_PATH + dirname)

        for (x, y, w, h) in faces:
            # Enregistrement du visage (eigenface) dans un dossier.
            resized_image = cv2.resize(image[y:(y+h), x:(x+w)], (100, 100))
            cv2.imwrite(EIGENFACES_PATH + dirname + '/' + str(i) + '.jpeg', resized_image)
            i += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

print('-' * 80)

######################################
# Création du csv pour le Classifier #
######################################

print("# Création du csv pour l'apprentissage.")
print('-' * 80)

CSV_FILE = 'faces.csv'
SEPARATOR = ";"
FACES_FOLDER = "eigenfaces/"

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

print('> Csv créé.')
print('-' * 80)

########################################
# Apprentissage et prédiction du sujet #
########################################

print("# Apprentissage des visages.")
print('-' * 80)

with open(CSV_FILE, 'r') as file:
    csv = file.readlines()

ratio = 0.1
random.shuffle(csv)
size = int(math.floor(ratio*len(csv)))
training_data, testing_data = csv[size:], csv[:size]

print("Nb. données d'entrainement data: " + str(len(training_data)))
print('Nb. données de test: ' + str(len(testing_data)))

print('-' * 80)

data = {}

for line in training_data:
    filename, label = line.strip().split(';')

    if label in data:
        numpy.append(data[int(label)], cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    else:
        data[int(label)] = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)

model = cv2.createEigenFaceRecognizer()
print("Label des visages:")
print(data.keys())
print('-' * 80)
model.train(data.values(), numpy.array(data.keys()))

error = 0
nb_elements = 0
for line in testing_data:
    nb_elements += 1
    filename, label =  line.strip().split(';')
    predicted_label = model.predict(cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    print '> Prediction: %(predicted)s  Réel: %(actual)s' %  {"predicted": predicted_label[0], "actual": label}
    if str(predicted_label[0]) != str(label):
        print('>> Erreur de prédiction !')
        error += 1
    else:
        print('>> Prédiction ok !')

print('-' * 80)
print("Taux d'erreur: " + str((error*100)//nb_elements) + '%.')
print("Pour avoir un taux d'erreur plus faible, il faut une base d'apprentissage plus conséquente.")
print('-' * 80)
