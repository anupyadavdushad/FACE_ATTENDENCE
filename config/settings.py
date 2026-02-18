import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "students")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
ATTENDANCE_LOG_DIR = os.path.join(BASE_DIR, "data", "attendance_logs")

EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.pkl")

IMAGE_SIZE = (160, 160)
MIN_IMAGES_REQUIRED = 10
