import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your trained model
model = load_model("resnet50_biopsy_final_model.h5")  # Adjust filename if needed
print("Model loaded successfully.")