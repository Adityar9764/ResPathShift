import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Dummy Cast class to handle unknown 'Cast' layer during load
class Cast(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# Load the model with custom object scope
model = load_model("resnet50_biopsy_model_clahe_finetuned.h5", custom_objects={'Cast': Cast})


# Define class names based on folder names
EXTERNAL_DATA_PATH = "external_dataset_2"
class_names = sorted(os.listdir(EXTERNAL_DATA_PATH))

# Set image size
IMAGE_SIZE = (224, 224)

# Store true and predicted labels
y_true = []
y_pred = []

print("Starting external dataset prediction...")

# Loop through each class
for label_idx, class_name in enumerate(class_names):
    class_folder = os.path.join(EXTERNAL_DATA_PATH, class_name)
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        try:
            # Preprocess the image
            img = image.load_img(img_path, target_size=IMAGE_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            y_true.append(label_idx)
            y_pred.append(predicted_class)

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt="d")
plt.title("Confusion Matrix on External Dataset")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()
