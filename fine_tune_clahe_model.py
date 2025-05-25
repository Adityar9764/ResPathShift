import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, Layer
import cv2

# === 🧠 Custom Cast layer if used previously ===
class Cast(Layer):
    def __init__(self, name="cast_custom", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# === ✅ Load Pretrained CLAHE Model ===
model = load_model("resnet50_biopsy_model_clahe_v2.h5", custom_objects={"Cast": Cast})
print("✅ Model loaded.")

# === 🧱 Add Dropout Layer (if not already present) ===
if not any(isinstance(layer, Dropout) for layer in model.layers):
    x = model.layers[-2].output
    x = Dropout(0.5)(x)
    x = model.layers[-1](x)
    model = Model(inputs=model.input, outputs=x)
    print("🔁 Dropout layer added.")

# === 🧪 Training Settings ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 6
FINE_TUNE_PATH = "balanced_finetune_set"  # This folder should match original structure

# === 🔁 Augmented Data Generator with strong transformations ===
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)

train_gen = datagen.flow_from_directory(
    FINE_TUNE_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    FINE_TUNE_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# === 🔓 Unfreeze fewer layers to avoid overfitting ===
model.trainable = True
for layer in model.layers[:-50]:  # Freeze all but last 50 layers
    layer.trainable = False
print("🔓 Unfrozen last 50 layers for fine-tuning.")

# === 🧠 Compile model with label smoothing ===
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

# === 🧠 Smart Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# === 🚀 Fine-tune the model ===
print("🚀 Starting fine-tuning...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === 💾 Save model ===
model.save("resnet50_biopsy_model_clahe_finetuned_final.h5")
print("✅ Fine-tuned model saved as 'resnet50_biopsy_model_clahe_finetuned_final.h5'")
