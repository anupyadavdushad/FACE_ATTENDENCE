import cv2
import numpy as np
from numpy.linalg import norm 
from keras_facenet import FaceNet

class FaceEmbedder:
    def __init__(self, model=None):
        """
        model: uses FaceNet from keras_facenet
        """
        if model is None:
            self.model = FaceNet()


    def preprocess(self, face_img):
        """
        Input: BGR face image (OpenCV)
        Output: normalized RGB tensor
        """

        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        img = img.astype("float32")
        img = (img - 127.5) / 128.0
        return img
    

    def get_embedding(self, face_img):
        """
        Returns L2-normalized embedding vector
        """
        preprocessed = self.preprocess(face_img)
        embedding = self.model.predict(preprocessed)[0]
        embedding = embedding / norm(embedding)
        
        return embedding 
    