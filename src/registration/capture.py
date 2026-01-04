import cv2
import os
import time
import numpy as np 
from mtcnn import MTCNN
from quality import FaceQualityChecker 

#------------CONFIG----------------------
USER_ID = "user_002"  #change per person
MAX_SAMPLES = 30
SAVE_INTERVAL = 0.5
MIN_FACE_SIZE = 100
BLUR_THRESHOLD = 100.0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_DIR = os.path.join(BASE_DIR,"data","raw",USER_ID)
os.makedirs(SAVE_DIR, exist_ok=True)

detector = MTCNN()
cap = cv2.VideoCapture(0)

saved_count = 0
last_save_time = 0

print("Starting face registration.....")
print("press 'q' to quit early")

while cap.isOpened and saved_count < MAX_SAMPLES:
    ret, frame = cap.read()
    
    if not ret: #Camera fails
        break
    
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if not faces:
        continue  # skip empty detection
    for face in faces:
        x, y, w, h = face["box"]
        x, y = max(0,x),max(0,y)

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        face_crop = frame[y:y+h,x:x+w]
        if face_crop.size == 0:
            continue

        #Add quality file funtions here 
        qc = FaceQualityChecker()
        usable, reason = qc.is_usable(face_crop, face.get("keypoints"))
        if not usable:
            print(reason)
            continue

        now = time.time()
        if now-last_save_time >= SAVE_INTERVAL:
            save_path = os.path.join(SAVE_DIR,f"{saved_count:03d}.jpg")
            ret = cv2.imwrite(save_path, face_crop)
            saved_count += 1
            last_save_time = now
            print(f"[SAVED] {saved_count}/{MAX_SAMPLES}")

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        
    cv2.putText(frame, f"Samples: {saved_count}/{MAX_SAMPLES}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("[DONE] Registration finished")
        