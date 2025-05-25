import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Set Dataset Path
DATASET_PATH = "archive/"
LUNG_PATH = os.path.join(DATASET_PATH, "lung_image_sets")
COLON_PATH = os.path.join(DATASET_PATH, "colon_image_sets")

# Check Data Structure
for folder in [LUNG_PATH, COLON_PATH]:
    print(f"Contents of {folder}:")
    print(os.listdir(folder))

# Function to Display Sample Images
def display_sample_images(class_path, num_samples=5):
    categories = os.listdir(class_path)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i in range(num_samples):
        category = np.random.choice(categories)
        img_path = os.path.join(class_path, category, np.random.choice(os.listdir(os.path.join(class_path, category))))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[i].imshow(img)
        axes[i].set_title(category)
        axes[i].axis("off")

    plt.show()

# Display images from both datasets
display_sample_images(LUNG_PATH)
display_sample_images(COLON_PATH)
