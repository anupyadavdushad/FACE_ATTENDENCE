import cv2
import numpy as np


class FaceQualityChecker:
    def __init__(
            self,
            min_size=100,
            blur_threshold=100.0,
            min_brightness=60,
            max_brightness=200,
            max_eye_tilt=10
    ):
        self.min_size = min_size
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_eye_tilt = max_eye_tilt

    def check_size(self, face):
        h, w = face.shape[:2]
        return h >= self.min_size and w >= self.min_size
    
    def check_blur(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score >= self.blur_threshold
    
    def check_brightness(self, face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        return self.min_brightness <= mean <= self.max_brightness
    
    def check_pose(self, landmarks):
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        eye_tilt = abs(left_eye[1]-right_eye[1])
        return eye_tilt <= self.max_eye_tilt
    
    def is_usable(self, face, landmarks=None):
        if not self.check_size(face):
            return False, "size"
        # if not self.check_blur(face):
            # return False, "blur"
        if not self.check_brightness(face):
            return False, "brigthness"
        if landmarks and not self.check_pose(landmarks):
            return False, "pose"
        return True, "ok"