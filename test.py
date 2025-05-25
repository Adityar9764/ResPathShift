# import tensorflow as tf
# print(tf.__version__)

# import pandas as pd
# print(pd.__version__)

# import numpy as np
# print(np.__version__)


import os

val_dir = "val_data/"
total_test_images = 0

for class_name in os.listdir(val_dir):
    class_path = os.path.join(val_dir, class_name)
    total_test_images += len(os.listdir(class_path))

print(f"Total test images: {total_test_images}")
