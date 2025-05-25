import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.layers import Layer


# === ✅ 1. Load the fine-tuned model (with Cast fix) ===
class Cast(Layer):
    def __init__(self, name="cast_custom", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

model = load_model("resnet50_biopsy_model_clahe_finetuned_v2.h5", custom_objects={'Cast': Cast})
print("✅ Model loaded.")


# === 1. Set path to your external dataset ===
EXTERNAL_DATASET_PATH = "external_dataset_2"  

# === 2. Set image size and batch size ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# === 3. Apply same preprocessing (e.g. CLAHE or just rescale) ===
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    EXTERNAL_DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # VERY IMPORTANT: keep order for accurate mapping
)


# Get true class names (label → class index mapping)
class_names = list(test_generator.class_indices.keys())

# Get paths to images
file_paths = test_generator.filepaths

# Get ground truth class indices
y_true = test_generator.classes

# Predict using the model
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)

print("✅ All required variables generated.")



# === Output folder to save misclassified images ===
MISCLASSIFIED_DIR = "misclassified_visuals"
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# Clear folder first if rerunning
for f in os.listdir(MISCLASSIFIED_DIR):
    shutil.rmtree(os.path.join(MISCLASSIFIED_DIR, f), ignore_errors=True)

# === Loop through misclassified images and save ===
for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
        true_label = class_names[y_true[i]]
        predicted_label = class_names[y_pred[i]]
        img_path = file_paths[i]
        
        # Create class-wise folder: e.g. lung_n_as_lung_aca
        subfolder = f"{true_label}_as_{predicted_label}"
        save_dir = os.path.join(MISCLASSIFIED_DIR, subfolder)
        os.makedirs(save_dir, exist_ok=True)

        # Copy image to appropriate folder
        shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))

print(f"✅ Misclassified images saved in: '{MISCLASSIFIED_DIR}'")
