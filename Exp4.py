import cv2
import os
import numpy as np
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, labels = [], []
label_names = []
for i, person in enumerate(os.listdir("dataset")):
    label_names.append(person)
    for img_name in os.listdir(f"dataset/{person}"):
        img = cv2.imread(f"dataset/{person}/{img_name}", cv2.IMREAD_GRAYSCALE)
        faces.append(cv2.resize(img, (160,160)))
        labels.append(i)
recognizer.train(np.array(faces), np.array(labels))
test_img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (160,160))
label, conf = recognizer.predict(test_img)
print("Predicted:", label_names[label])
