import cv2
import numpy as np
from numpy.linalg import norm
from keras_facenet import FaceNet

class FaceEmbedder:
    def __init__(self):
        self.model = FaceNet()

    def preprocess(self, img):
        # Force valid image
        if img is None:
            raise ValueError("Input image is None")

        # # Handle grayscale images
        # if len(img.shape) == 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # # Handle RGBA just in case
        # if img.shape[2] == 4:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (160, 160))

        return img

    def get_embedding(self, img):
        img = self.preprocess(img)

        # IMPORTANT: batch dimension
        batch = np.expand_dims(img, axis=0)  # (1, 160, 160, 3)

        embedding = self.model.embeddings(batch)[0]
        embedding = embedding / norm(embedding)

        return embedding
