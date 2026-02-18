import cv2
import numpy as np


class FaceQualityChecker:
    def __init__(self, blur_threshold=80):
        self.blur_threshold = blur_threshold

    def is_blurry(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def check(self, frame):
        if self.is_blurry(frame):
            return False, "Blurry Image"
        return True, "Good"
