# 📌 Face Recognition Attendance System

A real-time **Face Recognition Based Attendance System** built using **Python** and **OpenCV**, designed to automate attendance marking by detecting and recognizing faces through a camera feed. The system supports **new user registration**, **dataset creation**, and **attendance logging** with timestamp storage.

---

## 🚀 Features

- ✅ Real-time face detection using webcam
- ✅ Face recognition based identity matching
- ✅ Automatic attendance marking with date & time
- ✅ New user registration (dataset generation)
- ✅ Stores attendance records in structured format (CSV)
- ✅ Scalable design (supports multiple users)
- ✅ Simple and easy-to-run project structure

---

## 🧠 Project Workflow

1. **User Registration**
   - Captures multiple face images of a new user.
   - Stores them in a dataset directory.

2. **Model / Encoding Creation**
   - Generates face encodings for all stored faces.

3. **Real-Time Recognition**
   - Opens webcam feed.
   - Detects faces and compares encodings.

4. **Attendance Marking**
   - Marks recognized person’s attendance.
   - Saves name + timestamp in an attendance file.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** OpenCV, NumPy, Pandas  
- **Tools:** Git, GitHub, VS Code / Jupyter Notebook  
- **Concepts Used:** Face Detection, Face Recognition, Feature Extraction

---

## 📂 Folder Structure

```bash
Face-Attendance-System/
│
├── dataset/                  # Stores captured face images of registered users
│   ├── person1/
│   ├── person2/
│
├── attendance/               # Stores attendance logs
│   ├── attendance.csv
│
├── capture.py                # Script for capturing images and adding new users
├── recognition.py            # Main face recognition + attendance marking script
├── utils.py                  # Helper functions (optional)
├── requirements.txt          # Required dependencies
└── README.md                 # Project documentation
