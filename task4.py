# task4.py
# Hand Gesture Recognition using SVM (Fixed for Leap Dataset)

import os
import cv2
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "leapGestRecog"
IMG_SIZE = 64
SAMPLES_PER_GESTURE = 50  # safe limit

# -----------------------------
# Load Images and Labels
# -----------------------------
X = []
y = []

gesture_labels = {}
label_count = 0

for subject in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)

        if gesture not in gesture_labels:
            gesture_labels[gesture] = label_count
            label_count += 1

        files = os.listdir(gesture_path)[:SAMPLES_PER_GESTURE]

        for file in tqdm(files, desc=f"Loading {gesture}"):
            img_path = os.path.join(gesture_path, file)

            img = cv2.imread(img_path)
            if img is None:
                continue  # skip unreadable files safely

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            X.append(img.flatten())
            y.append(gesture_labels[gesture])

X = np.array(X) / 255.0
y = np.array(y)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train SVM Model
# -----------------------------
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
