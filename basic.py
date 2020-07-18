import cv2
import face_recognition

# finding image original
tomOriginal = face_recognition.load_image_file('imageOriginal/tom-original.jpeg')
tomOriginal = cv2.cvtColor(tomOriginal, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(tomOriginal)[0]
encodeTomOriginal = face_recognition.face_encodings(tomOriginal)[0]
cv2.rectangle(tomOriginal, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), color=(255, 0, 0), thickness=2)
cv2.imshow('Tom Cruise', tomOriginal)

tomTest = face_recognition.load_image_file('imageTest/tom-test-2.jpg')
tomTest = cv2.cvtColor(tomTest, cv2.COLOR_BGR2RGB)

tomTestLoc = face_recognition.face_locations(tomTest)[0]
encodeTomTest = face_recognition.face_encodings(tomTest)[0]
cv2.rectangle(tomTest, (tomTestLoc[3], tomTestLoc[0]), (tomTestLoc[1], tomTestLoc[2]), color=(255, 0, 0), thickness=2)
cv2.imshow('Tom Cruise test', tomTest)

results = face_recognition.compare_faces([encodeTomOriginal], encodeTomTest)
face_distance = face_recognition.face_distance([encodeTomOriginal], encodeTomTest)
print(f"{results[0]}, {round(face_distance[0], 2)}")

# text isn't showing
cv2.putText(tomOriginal, f"{results}, {round(face_distance[0], 2)}",
            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 4,
            (255, 255, 255), 2, cv2.LINE_AA)


# cv2.rectangle(image, (x, y), (w + h), color=(255, 0, 0), thickness=2)
cv2.waitKey(0)
