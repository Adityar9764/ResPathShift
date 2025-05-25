import os
import shutil

# Define original dataset structure
ORIG_DATASET_PATH = "/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/"
WORKING_DIR = "/kaggle/working"
TRAIN_DIR = os.path.join(WORKING_DIR, "train_data")
VAL_DIR = os.path.join(WORKING_DIR, "val_data")

# Clean and recreate directories
for dir_path in [TRAIN_DIR, VAL_DIR]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# Navigate into subfolders: lung_image_sets/ and colon_image_sets/
for subset in ["lung_image_sets", "colon_image_sets"]:
    subset_path = os.path.join(ORIG_DATASET_PATH, subset)
    
    for category in os.listdir(subset_path):
        class_path = os.path.join(subset_path, category)
        
        if os.path.isdir(class_path):
            images = sorted(os.listdir(class_path))  # Ensure deterministic order
            total = len(images)
            split_index = int(total * 0.8)
            
            train_images = images[:split_index]
            val_images = images[split_index:]
            
            # Create target class folders in train and val dirs
            os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
            os.makedirs(os.path.join(VAL_DIR, category), exist_ok=True)
            
            for img in train_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, category, img))
            for img in val_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(VAL_DIR, category, img))

print("✅ Dataset successfully split (first 80% train, last 20% val)")





import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



# Data generators without validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Enable GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU.")

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Build model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# Callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[tensorboard_callback]
)

# Save new model
model.save("/kaggle/working/resnet50_biopsy_no_shuffle_model.h5")
print("Model training completed and saved as 'resnet50_biopsy_no_shuffle_model.h5'.")