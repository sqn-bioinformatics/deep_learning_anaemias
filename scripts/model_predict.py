import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from datetime import datetime

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

# Define the path to the directory containing the images
image_dir = Path(
    "/home/t.afanasyeva/research_storage/Processing/Lab - Van Dam/datasets/srf_anaemias/CytPix/processed/23-714262"
)

# Get a list of all .png files in the directory
image_files = list(image_dir.glob("*.png"))

# Select 100 random images
random_images = random.sample(image_files, 100)
with tf.device("/CPU:0"):

    # Load and resize images as float tensors
    image_tensors = [
        tf.image.resize(
            image.img_to_array(image.load_img(img_path)).astype(np.float32), (224, 224)
        )
        for img_path in random_images
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
    print("Class names:", class_names)

    num_images = 9
    num_batches = len(image_tensors) // num_images

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
