from encang.FaceService import FaceService
import cv2


path_image_1 = 'imageOriginal/tom-original.jpeg'
image1 = FaceService(path_image_1)
img = image1.faceRGB
faceLoc = image1.faceLocation(img)
image1.faceBorderRectangle(img, faceLoc)
image1.showFace('', img)


cv2.waitKey(0)
