import cv2
import numpy as np
import os

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Part 1: Create fisherRecognizer
print('Recognizing Face. Please ensure sufficient light conditions...')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            image = cv2.imread(path, 0)
            if image is not None:
                resized_image = cv2.resize(image, (130, 100))
                images.append(resized_image)
                labels.append(int(label))
        id += 1

# Create a Numpy array from the two lists above
(images, labels) = [np.array(lis) for lis in [images, labels]]
labels = labels.astype(np.int32)  # Ensure labels are of type int32

# OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (130, 100))
        
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 500:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(im, 'not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    cv2.imshow('OpenCV', im)
    
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
