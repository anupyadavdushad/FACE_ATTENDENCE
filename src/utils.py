import os
import pickle
import pandas as pd
from datetime import datetime


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pickle(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_attendance_csv(log_dir, name, reg_no):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"
    filepath = os.path.join(log_dir, filename)

    record = {
        "Name": name,
        "RegNo": reg_no,
        "Time": datetime.now().strftime("%H:%M:%S")
    }

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if reg_no in df["RegNo"].astype(str).values:
            return filepath
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(filepath, index=False)
    return filepath
