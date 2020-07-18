import cv2
import face_recognition
import os
import numpy as np

path = 'ImageAttendants'
images = []
name = []
pathImage = os.listdir(path)

for image in pathImage:
    currentImage = cv2.imread(f'{path}/{image}')
    images.append(currentImage)
    name.append(os.path.splitext(image)[0])


def find_encoding(images):
    image_encoding = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        image_encoding.append(encode)
    return image_encoding


encodeListKnow = find_encoding(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
while True:
    success, img = cap.read()
    imageSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imageSmall = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2RGB)
    faceCurrent = face_recognition.face_locations(imageSmall)
    encode = face_recognition.face_encodings(imageSmall, faceCurrent)

    for encodeFace, faceLoc in zip(encode, faceCurrent):
        matcher = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matcher[matchIndex]:
            user = name[matchIndex].upper()
            x, y, w, h = faceLoc
            x, y, w, h = x * 4, y * 4, w * 4, h * 4
            cv2.rectangle(imageSmall,
                          (x, y,),
                          (w, h,),
                          color=(255, 0, 0),
                          thickness=2)
            cv2.rectangle(imageSmall,
                          (x, h-20,),
                          (w, h,),
                          (255, 0, 0),
                          cv2.FILLED)
            cv2.putText(imageSmall, f"{user}",
                        (x, h), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 255), 2)
            print(user)
        else:
            print('Unknown')

    cv2.imshow('Web cam', imageSmall)
    cv2.waitKey(1)
