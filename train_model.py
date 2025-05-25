import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime


# Set GPU usage to full capacity
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print("GPU is being used for training")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, training on CPU instead.")


# Enable mixed precision (faster computation on GPU)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Dataset Path
DATASET_PATH = "archive/"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12  # As per your requirement

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% Train, 20% Test
)

# Load Training Data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load Validation Data
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load Pre-trained ResNet50 (Without Top Layers)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers (to use pre-trained features)
base_model.trainable = False  

# Add Custom Classification Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)  # Reduce overfitting
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(train_generator.num_classes, activation="softmax")(x)  # Multi-class classification

# Create Model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Model Summary
model.summary()

# Callbacks (For Better Training)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train Model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[tensorboard_callback]
)

# Save Model
model.save("resnet50_biopsy_model.h5")

print("Model training completed and saved as 'resnet50_biopsy_model.h5'.")
