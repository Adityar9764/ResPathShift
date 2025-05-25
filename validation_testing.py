import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_custom_objects


# Define a dummy Cast layer to bypass the unknown layer issue
class CustomCastLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

get_custom_objects().update({'Cast': CustomCastLayer})


# Load your trained model
model = load_model("resnet50_biopsy_final_model.h5")  # Adjust filename if needed
print("Model loaded successfully.")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "val_data/",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important for correct label-order matching
)


# Predict class probabilities
pred_probs = model.predict(test_generator)

# Get predicted class indices
y_pred = np.argmax(pred_probs, axis=1)

# Get true class indices
y_true = test_generator.classes

# Map indices back to class names
class_names = list(test_generator.class_indices.keys())


from sklearn.metrics import classification_report, confusion_matrix

# Print classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


## Accuracy Calculation
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")
