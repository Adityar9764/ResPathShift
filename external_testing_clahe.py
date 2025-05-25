import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Dummy Cast class
class Cast(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Load the CLAHE-preprocessed model
model = load_model("resnet50_biopsy_model_clahe_v2.h5", custom_objects={'Cast': Cast})
IMAGE_SIZE = (224, 224)
EXTERNAL_DATA_PATH = "external_dataset_2"
class_names = sorted(os.listdir(EXTERNAL_DATA_PATH))

y_true = []
y_pred = []

# --- Preprocess with CLAHE ---
def preprocess_with_clahe(image_array):
    lab = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return rgb / 255.0

print("Starting external dataset prediction...")

for label_idx, class_name in enumerate(class_names):
    class_folder = os.path.join(EXTERNAL_DATA_PATH, class_name)
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        try:
            img = image.load_img(img_path, target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img)
            img_array = preprocess_with_clahe(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            y_true.append(label_idx)
            y_pred.append(predicted_class)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")



correct_preds = np.sum(np.array(y_true) == np.array(y_pred))
total_preds = len(y_true)
raw_accuracy = correct_preds / total_preds

final_accuracy = raw_accuracy  

# --- Report ---
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt="d")
plt.title("Confusion Matrix on External Dataset")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()



print(f"\nAdjusted Accuracy on External Dataset: {final_accuracy * 100:.2f}%")
