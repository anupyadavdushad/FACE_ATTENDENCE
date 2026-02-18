import cv2
import numpy as np
from config.settings import IMAGE_SIZE


class FaceEmbedder:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def get_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, IMAGE_SIZE)
        return face

    def get_embedding(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        embedding = cv2.resize(gray, (32, 32)).flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
