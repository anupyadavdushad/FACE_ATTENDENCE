import cv2
import os
import time

from config.settings import RAW_DATA_DIR
from src.utils import ensure_dir


def capture_frames_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        yield frame

    cap.release()


def save_face_image(frame, save_path, count):
    img_path = os.path.join(save_path, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    return img_path


def create_student_folder(name, reg_no):
    folder_name = f"{reg_no}_{name}"
    save_path = os.path.join(RAW_DATA_DIR, folder_name)
    ensure_dir(save_path)
    return save_path
