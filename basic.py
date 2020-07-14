import cv2
import face_recognition

# finding image
tomOriginal = face_recognition.load_image_file('imageOriginal/tom-original.jpeg')
tomOriginal = cv2.cvtColor(tomOriginal, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(tomOriginal)[0]
encodeElon = face_recognition.face_encodings(tomOriginal)[0]
# cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

cv2.rectangle(tomOriginal, (faceLoc[3], faceLoc[1]), (faceLoc[1] * 0, faceLoc[2] * 0), color=(255, 0, 0), thickness=2)

print(faceLoc)
print(faceLoc[0])
print(faceLoc[1])
print(faceLoc[2])
print(faceLoc[3])

cv2.imshow('Tom Cruise', tomOriginal)
cv2.waitKey(0)
