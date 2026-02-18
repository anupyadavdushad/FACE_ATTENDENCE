import streamlit as st
import os
import time
import pandas as pd
import cv2
import re

from config.settings import ATTENDANCE_LOG_DIR, RAW_DATA_DIR
from src.utils import ensure_dir
from src.create_embedding import create_embeddings
from src.recognition.matcher import match_face, mark_attendance


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def clean_text(txt: str):
    txt = txt.strip()
    txt = re.sub(r"[^\w\s-]", "", txt)   # remove special chars
    txt = txt.replace(" ", "_")          # spaces -> underscore
    return txt


def create_student_folder(name, reg_no):
    name = clean_text(name)
    reg_no = clean_text(reg_no)

    folder_name = f"{reg_no}_{name}"
    save_path = os.path.join(RAW_DATA_DIR, folder_name)
    ensure_dir(save_path)
    return save_path


def save_face_image(frame, save_path, count):
    img_path = os.path.join(save_path, f"{count}.jpg")
    cv2.imwrite(img_path, frame)
    return img_path


# ----------------------------------------------------
# Streamlit Setup
# ----------------------------------------------------
st.set_page_config(page_title="Face Attendance System", page_icon="📸", layout="wide")

st.title("📸 Face Attendance System")
st.write("Register Students, Generate Embeddings, and Mark Attendance using Live Camera.")


menu = st.sidebar.radio(
    "Select Option",
    ["🏠 Home", "📝 Register Student", "🧠 Generate Embeddings", "✅ Mark Attendance", "📄 View Logs"]
)


# ----------------------------------------------------
# HOME
# ----------------------------------------------------
if menu == "🏠 Home":
    st.header("🏠 Home")
    st.success("System Ready.")
    st.info("Use the sidebar to navigate.")


# ----------------------------------------------------
# REGISTER STUDENT
# ----------------------------------------------------
elif menu == "📝 Register Student":
    st.header("📝 Register New Student (Live Capture)")

    name = st.text_input("Student Name")
    reg_no = st.text_input("Registration Number")

    num_images = st.slider("Number of images to capture", 5, 50, 10)

    col1, col2 = st.columns(2)

    if "capturing" not in st.session_state:
        st.session_state.capturing = False

    if "captured_count" not in st.session_state:
        st.session_state.captured_count = 0

    if "capture_logs" not in st.session_state:
        st.session_state.capture_logs = []

    if "captured_frames" not in st.session_state:
        st.session_state.captured_frames = []

    with col1:
        start_btn = st.button("📷 Start Capture")

    with col2:
        stop_btn = st.button("🛑 Stop Capture")

    if start_btn:
        if name.strip() == "" or reg_no.strip() == "":
            st.error("Name and Registration Number cannot be empty.")
        else:
            st.session_state.capturing = True
            st.session_state.captured_count = 0
            st.session_state.captured_frames = []
            st.session_state.capture_logs = []
            st.session_state.capture_logs.append("Camera started...")

    if stop_btn:
        st.session_state.capturing = False
        st.session_state.capture_logs.append("Capture stopped by user.")

    camera_placeholder = st.empty()
    progress_bar = st.progress(0)
    count_text = st.empty()

    st.subheader("📟 Capture Logs")
    log_box = st.empty()

    st.subheader("🖼 Captured Images Preview")
    image_box = st.empty()

    if st.session_state.capturing:

        save_path = create_student_folder(name, reg_no)
        st.session_state.capture_logs.append(f"Saving images to: {save_path}")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not accessible.")
            st.session_state.capturing = False

        while st.session_state.capturing and st.session_state.captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            # Display live camera feed
            camera_placeholder.image(frame, channels="BGR", use_container_width=True)

            # auto capture every 0.7 sec
            time.sleep(0.7)

            img_path = save_face_image(frame, save_path, st.session_state.captured_count)

            st.session_state.captured_frames.append(frame.copy())
            st.session_state.captured_count += 1

            st.session_state.capture_logs.append(
                f"[{st.session_state.captured_count}/{num_images}] Captured: {img_path}"
            )

            # Update progress
            progress = st.session_state.captured_count / num_images
            progress_bar.progress(progress)

            count_text.markdown(
                f"### ✅ Captured: `{st.session_state.captured_count} / {num_images}`"
            )

            # Update logs
            log_box.code("\n".join(st.session_state.capture_logs[-25:]), language="text")

            # Preview last 6 images
            preview_images = st.session_state.captured_frames[-6:]
            image_box.image(preview_images, channels="BGR", width=150)

        cap.release()

        if st.session_state.captured_count >= num_images:
            st.session_state.capturing = False
            st.success("✅ Capture Completed Successfully!")
            st.balloons()


# ----------------------------------------------------
# GENERATE EMBEDDINGS
# ----------------------------------------------------
elif menu == "🧠 Generate Embeddings":
    st.header("🧠 Generate Embeddings")

    st.warning("Run this after registering new students.")

    if st.button("⚡ Generate Embeddings Now"):
        try:
            path = create_embeddings()
            st.success(f"✅ Embeddings created successfully: {path}")
        except Exception as e:
            st.error(f"Error while generating embeddings: {e}")


# ----------------------------------------------------
# MARK ATTENDANCE
# ----------------------------------------------------
elif menu == "✅ Mark Attendance":
    st.header("✅ Mark Attendance (Live Camera Mode)")

    threshold = st.slider("Matching Threshold", 0.1, 1.0, 0.6, 0.01)

    col1, col2 = st.columns(2)

    if "attendance_running" not in st.session_state:
        st.session_state.attendance_running = False

    if "attendance_logs" not in st.session_state:
        st.session_state.attendance_logs = []

    if "attendance_marked" not in st.session_state:
        st.session_state.attendance_marked = set()

    with col1:
        start_btn = st.button("🎥 Start Attendance")

    with col2:
        stop_btn = st.button("🛑 Stop Attendance")

    if start_btn:
        st.session_state.attendance_running = True
        st.session_state.attendance_logs = []
        st.session_state.attendance_marked = set()
        st.session_state.attendance_logs.append("Attendance camera started...")

    if stop_btn:
        st.session_state.attendance_running = False
        st.session_state.attendance_logs.append("Attendance stopped by user.")

    camera_placeholder = st.empty()

    st.subheader("📌 Status")
    status_text = st.empty()

    st.subheader("📟 Attendance Logs")
    log_display = st.empty()

    st.subheader("📊 Attendance Marked Count")
    count_display = st.empty()

    if st.session_state.attendance_running:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not accessible.")
            st.session_state.attendance_running = False

        while st.session_state.attendance_running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            status, best_match, score = match_face(frame, threshold=threshold)

            if status == "NO_FACE":
                status_text.warning("No face detected...")

            elif status == "UNKNOWN":
                status_text.error(f"Unknown Face (score={score:.2f})")

            elif status == "MATCH":
                name = best_match["name"]
                reg_no = best_match["reg_no"]

                if reg_no not in st.session_state.attendance_marked:
                    mark_attendance(best_match)
                    st.session_state.attendance_marked.add(reg_no)

                    st.session_state.attendance_logs.append(
                        f"✅ Attendance marked: {name} ({reg_no}) score={score:.2f}"
                    )

                status_text.success(f"Matched: {name} ({reg_no}) score={score:.2f}")

            # Show camera feed
            camera_placeholder.image(frame, channels="BGR", use_container_width=True)

            # Update logs
            log_display.code("\n".join(st.session_state.attendance_logs[-25:]), language="text")

            # Update count
            count_display.metric("Marked Students", len(st.session_state.attendance_marked))

            time.sleep(0.25)

        cap.release()


# ----------------------------------------------------
# VIEW LOGS
# ----------------------------------------------------
elif menu == "📄 View Logs":
    st.header("📄 Attendance Logs")

    ensure_dir(ATTENDANCE_LOG_DIR)

    files = [f for f in os.listdir(ATTENDANCE_LOG_DIR) if f.endswith(".csv")]

    if len(files) == 0:
        st.warning("No attendance logs found yet.")
    else:
        selected_file = st.selectbox("Select Attendance Log File", files)

        file_path = os.path.join(ATTENDANCE_LOG_DIR, selected_file)
        df = pd.read_csv(file_path)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="⬇ Download Log CSV",
            data=df.to_csv(index=False),
            file_name=selected_file,
            mime="text/csv"
        )
