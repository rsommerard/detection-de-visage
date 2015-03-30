# coding: utf-8
import sys, os
import cv2

def detecte_visages(image, image_out, show = False):
    # on charge l'image en mémoire
    img = cv2.imread(image)
    # on charge le modèle de détection des visages
    face_model = cv2.CascadeClassifier("opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # détection du ou des visages
    faces = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    # on place un cadre autour des visages
    print ("nombre de visages", len(faces), "dimension de l'image", img.shape, "image", image)
    for face in faces:
        cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[0] + face[3]), (255, 0, 0), 3)

    # on sauvegarde le résultat final
    cv2.imwrite(image_out, img)

    # pour voir l'image, presser ESC pour sortir
    if show :
        cv2.imshow("visage",img)
        if cv2.waitKey(5000) == 27: cv2.destroyWindow("visage")

if __name__ == "__main__":
    # applique
    for file in os.listdir(".") :
        if file.startswith("visage") : continue # déjà traité
        if os.path.splitext(file)[-1].lower() in [".jpg", ".jpeg", ".png" ] :
            detecte_visages (file, "visage_" + file, True)


"""import cv2
import sys
# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
gray,
scaleFactor=1.1,
minNeighbors=5,
minSize=(30, 30),
flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
print "Found {0} faces!".format(len(faces))
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow("Faces found", image)
  cv2.waitKey(0)"""
