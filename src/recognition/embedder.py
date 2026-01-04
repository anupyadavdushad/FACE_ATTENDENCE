import cv2
import numpy as np
from numpy.linalg import norm 

class FaceEmbedder:
    def __init__(self, model):
        """
        model: a loaded face embedding model
        example: FaceNet, InsighFace, etc.
        """

        self.model = model 

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
    