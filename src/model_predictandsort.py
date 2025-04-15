import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from datetime import datetime
import pickle

print(tf.__version__)
tf.random.set_seed(42)

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)
plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

# # Define the path to the directory containing the images
# image_dir = Path(
#     "/home/t.afanasyeva/research_storage/Processing/Lab - Van Dam/datasets/srf_anaemias/CytPix/processed/"
# )
# # Get a list of all .png files in the directory and subdirectories that do not have the postfix _sorted
# image_files = [
#     f
#     for f in tqdm(image_dir.rglob("*.png"), desc="Scanning images")
#     if not f.stem.endswith("_sorted")
# ]

# print("Number of images:", len(image_files))

# # Define the path to the new directory containing the images
# new_image_dir = Path("/home/t.afanasyeva/deep_learning_anaemias/resources/train")
# # Get a list of all .png files in the new directory and subdirectories
# new_image_files = list(tqdm(new_image_dir.rglob("*.png"), desc="Scanning new images"))

# print("Number of new images:", len(new_image_files))

# # Extract the names of the files in new_image_files
# new_image_names = {f.name for f in new_image_files}

# # Filter image_files to only include files that do not share a name with new_image_files
# image_files = [
#     f
#     for f in tqdm(image_files, desc="Filtering images")
#     if f.name not in new_image_names
# ]

# print("Number of images after removal:", len(image_files))

# Define the path to save the image_files list
pickle_path = Path("/home/t.afanasyeva/deep_learning_anaemias/output/to_sort.pkl")

# # Save the image_files list to a pickle file
# with open(pickle_path, "wb") as f:
#     pickle.dump(image_files, f)

# print(f"Image files list saved to {pickle_path}")

# Load the image_files list from the pickle file
with open(pickle_path, "rb") as f:
    image_files = pickle.load(f)

print(f"Loaded {len(image_files)} images from {pickle_path}")


# Select 1000 random images
random_images = random.sample(image_files, 3000)
with tf.device("/CPU:0"):

    # Load and resize images as float tensors
    image_tensors = [
        tf.image.resize(
            image.img_to_array(image.load_img(img_path)).astype(np.float32), (224, 224)
        )
        for img_path in tqdm(random_images, desc="Loading and resizing images")
    ]

    # Create a TensorFlow dataset from the image tensors
    dataset = tf.data.Dataset.from_tensor_slices(image_tensors)

    # Batch the dataset
    batch_size = 20
    batched_dataset = dataset.batch(batch_size).prefetch(1)

    model = tf.keras.models.load_model(
        "/home/t.afanasyeva/deep_learning_anaemias/output/250208_cytpix/250208_EfficientNetB0_v2.keras"
    )

    y_pred = model.predict(batched_dataset)

    class_names = ["discocyte", "echinocyte", "granular", "holly_leaf", "sickle"]

    num_images = 9
    num_batches = 20

    for batch in range(num_batches):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for i, ax in enumerate(axes.flat):
            img_index = batch * num_images + i
            if img_index < len(image_tensors):
                img = image_tensors[img_index].numpy().astype(np.uint8)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(
                    f"{class_names[np.argmax(y_pred[img_index])]}", fontsize=16
                )

        plt.tight_layout()
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        plt.savefig(
            f"/home/t.afanasyeva/deep_learning_anaemias/output/250208_cytpix/{timestamp}_predicted_images_batch_{batch + 1}.png"
        )
        plt.show()
        plt.close()

    # Create the sorted directory if it doesn't exist
    sorted_dir = Path("/home/t.afanasyeva/deep_learning_anaemias/output/250208_sorted")
    sorted_dir.mkdir(parents=True, exist_ok=True)

    # Create directories for all class names
    for class_name in class_names:
        class_dir = sorted_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over the predictions and save the images in the corresponding folders
    for img_path, pred in zip(random_images, y_pred):
        predicted_class = class_names[np.argmax(pred)]
        if predicted_class != "discocyte":
            img = image.load_img(img_path)
            img.save(sorted_dir / predicted_class / img_path.name)
