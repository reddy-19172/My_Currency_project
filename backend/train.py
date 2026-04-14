import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

def preprocess(path):
    img = cv2.imread(path)

    if img is None:
        print("❌ Skipping bad file:", path)
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    gray = gray.reshape(IMG_SIZE, IMG_SIZE, 1)

    return gray

X = []
y = []

DATASET_PATH = "dataset"

for label, folder in enumerate(["fake", "real"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        # Skip non-image files
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            print("⚠ Skipping non-image:", file)
            continue

        img = preprocess(path)

        if img is not None:
            X.append(img)
            y.append(label)

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("✅ Total images loaded:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

# ❌ If dataset empty → stop
if len(X) == 0:
    print("❌ ERROR: No valid images found in dataset")
    exit()

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    run_eagerly=True   # 🔥 important fix
)

model.fit(X, y, epochs=5)

model.save("model.h5")

print("✅ model.h5 created successfully")