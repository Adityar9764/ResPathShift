import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --------- Paths to Training and External Images ---------
train_class_folder = r"D:\MiniProject\archive\lung_image_sets\lung_aca"  # Change if needed

external_base = r"D:\MiniProject\external_data"
external_class_folders = ['aca_bd', 'scc_pd', 'nor']  # Pick any 2–3 folders from your external data

# --------- Number of Images to Show from Each ---------
num_images = 3

# --------- Helper Function ---------
def plot_images(title, image_paths, row_index, total_rows):
    for i, img_path in enumerate(image_paths):
        plt.subplot(total_rows, num_images, row_index * num_images + i + 1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
        if row_index == 0:
            plt.title(f"{title} {i+1}")

# --------- Load Random Training Images ---------
train_images = random.sample(os.listdir(train_class_folder), num_images)
train_image_paths = [os.path.join(train_class_folder, img) for img in train_images]

# --------- Load Random External Images ---------
external_image_sets = []
for folder in external_class_folders:
    full_path = os.path.join(external_base, folder)
    images = random.sample(os.listdir(full_path), num_images)
    image_paths = [os.path.join(full_path, img) for img in images]
    external_image_sets.append((folder, image_paths))

# --------- Plotting ---------
total_rows = 1 + len(external_image_sets)
plt.figure(figsize=(15, 3 * total_rows))

# Plot training images
plot_images("Train", train_image_paths, row_index=0, total_rows=total_rows)

# Plot external images
for i, (label, img_paths) in enumerate(external_image_sets):
    plot_images(label, img_paths, row_index=i+1, total_rows=total_rows)

plt.tight_layout()
plt.show()
