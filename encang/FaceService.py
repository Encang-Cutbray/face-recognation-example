import cv2
import face_recognition


class FaceService:

    def __init__(self, path_image: str):
        self.path_image = path_image

    @property
    def faceRGB(self):
        return self.__imageToRGB()

    def load_image(self):
        return face_recognition.load_image_file(self.path_image)

    def __imageToRGB(self):
        image = self.load_image()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def faceLocation(self, img_rgb):
        return face_recognition.face_locations(img_rgb)[0]

    def faceEncoding(self, img_rgb):
        return face_recognition.face_encodings(img_rgb)[0]

    def faceBorderRectangle(self, img_rgb, face_location):
        cv2.rectangle(img_rgb, (face_location[3], face_location[0]),
                      (face_location[1], face_location[2]),
                      color=(255, 0, 0),
                      thickness=2)

    def showFace(self, name: str, img_rgb):
        cv2.imshow(name, img_rgb)
