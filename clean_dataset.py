import os
import cv2

DATASET_PATH = "archive/"
IMAGE_SETS = ["lung_image_sets", "colon_image_sets"]

# Function to check and remove corrupted images
def remove_corrupt_images(base_path):
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Removing corrupted image: {img_path}")
                        os.remove(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    os.remove(img_path)

# Run for both lung and colon datasets
for dataset in IMAGE_SETS:
    remove_corrupt_images(os.path.join(DATASET_PATH, dataset))

print("Dataset cleaning completed.")
