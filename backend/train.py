import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128
DATASET_PATH = "dataset"

# 🔹 Preprocess image
def preprocess(path):
    img = cv2.imread(path)

    if img is None:
        print("❌ Skipping:", path)
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = gray / 255.0
    gray = gray.reshape(IMG_SIZE, IMG_SIZE, 1)

    return gray

# 🔹 Load dataset
X = []
y = []

for label, folder in enumerate(["fake", "real"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = preprocess(path)

        if img is not None:
            X.append(img)
            y.append(label)

# 🔹 Convert to numpy
X = np.array(X)
y = np.array(y)

print("Total images:", len(X))

# ❌ Stop if empty
if len(X) == 0:
    print("Dataset empty ❌")
    exit()

# SHUFFLE (IMPORTANT FIX)
X, y = shuffle(X, y)

# 🔹 Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 🔹 Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#  TRAIN (IMPROVED)
model.fit(
    X, y,
    epochs=20,
    batch_size=8,
    validation_split=0.2
)

# 🔹 Save model
model.save("model.h5")

print("✅ Model trained & saved successfully")