import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras

from keras.models import Sequential

from keras.layers import (
    RandomRotation,
    RandomTranslation,
    RandomFlip,
    RandomContrast,
    Dense,
    ReLU,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from keras.regularizers import L2

from keras.losses import CategoricalCrossentropy
import pickle
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    ConfusionMatrixDisplay,
)

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

print(tf.__version__)
keras.backend.clear_session()
print(tf.config.list_physical_devices("GPU"))
tf.random.set_seed(42)

path_out = (
    Path(__file__).parent.parent / f"output/{datetime.now().strftime('%y%m%d')}_output"
)
path_out.mkdir(parents=True, exist_ok=True)


# def get_confusion_matrix(model_name, y_test, y_pred, class_names, path_out):

#     _, ax = plt.subplots(figsize=(8, 8))
#     cm = ConfusionMatrixDisplay.from_predictions(
#         y_test,
#         y_pred,
#         ax=ax,
#         xticks_rotation="vertical",
#         colorbar=False,
#         normalize="true",
#         display_labels=class_names,
#     )

#     plt.rc("font", size=12)
#     ax.set_title(f"Confusion Matrix {model_name}")
#     plt.savefig(path_out / f"confusion_matrix_{model_name}.png")


def plot_history(model_name, history, metrics, path_out):
    sns.lineplot(data=history[metrics[0]], label=metrics[0])
    sns.lineplot(data=history[metrics[1]], label=metrics[1])
    plt.xlabel("epochs")
    plt.ylabel("metric")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(path_out / f"{model_name}_{metrics}_history.png", bbox_inches="tight")
    plt.close()


# Load data
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = 224


def expand_ds(train_ds):
    """
    Expands a training dataset by applying a series of data augmentation transformations.

    Args:
        train_ds: A TensorFlow dataset containing training data.

    Returns:
        A TensorFlow dataset with augmented data, interleaved with the original dataset.
    """
    data_augmentation_list = [
        Sequential([RandomRotation(factor=0.15)]),
        Sequential([RandomTranslation(height_factor=0.1, width_factor=0.1)]),
        Sequential([RandomFlip()]),
        Sequential([RandomContrast(factor=0.1)]),
    ]

    ds_list = [
        train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        for data_augmentation in data_augmentation_list
    ]
    ds_list.append(train_ds)
    ds = tf.data.Dataset.from_tensor_slices(ds_list)
    train_ds = ds.interleave(
        lambda x: x,
        cycle_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return train_ds


def add_source_channel(image_tensor, source_id):
    """
    Adds a channel to the image that encodes the source dataset

    Args:
        image_tensor: A tensor of shape [H, W, C]
        source_id: The dataset identifier value to fill the new channel with

    Returns:
        A tensor with an additional channel containing the source_id value
    """

    # Assuming [H, W, C] format (TensorFlow standard)
    H, W, C = image_tensor.shape

    source_channel = tf.ones((H, W, 1), dtype=image_tensor.dtype) * source_id
    augmented_tensor = tf.concat([image_tensor, source_channel], axis=2)

    return augmented_tensor


def add_source_to_dataset(image, label, source_id):
    image_with_source = add_source_channel(image, source_id)
    return image_with_source, label


def learning_rate_schedule(epoch, lr):
    if epoch < 5:
        return (lr * tf.math.exp(0.5)).numpy()
    if epoch < 15:
        return lr
    else:
        return (lr * tf.math.exp(-0.1)).numpy()


def simplified_expand_ds(train_ds):
    """Simplified version with fewer augmentations for testing"""
    data_augmentation = Sequential([RandomFlip()])
    return train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )


def main():

    path_in = Path(__file__).parent.parent / "resources/imagestream"
    train_ds_im, test_ds_im = keras.utils.image_dataset_from_directory(
        path_in,
        labels="inferred",
        label_mode="categorical",
        class_names=[
            "discocyte",
            "holly_leaf",
            "granular",
            "sickle",
            "echinocyte",
        ],
        color_mode="grayscale",
        batch_size=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=93,
        validation_split=0.2,
        subset="both",
        data_format="channels_last",
        verbose=True,
    )
    # class_names = test_ds_im.class_names

    train_ds_im, test_ds_im = [
        ds.map(
            lambda x, y: (tf.image.grayscale_to_rgb(x), y), num_parallel_calls=AUTOTUNE
        )
        for ds in (train_ds_im, test_ds_im)
    ]
    train_ds_im = expand_ds(train_ds_im)

    train_ds_im, test_ds_im = [
        ds.map(
            lambda x, y: (add_source_to_dataset(x, y, 1)), num_parallel_calls=AUTOTUNE
        )
        for ds in (train_ds_im, test_ds_im)
    ]

    path_in_cp = Path(__file__).parent.parent / "resources/cytpix/augmented"

    train_ds_cp, test_ds_cp = keras.utils.image_dataset_from_directory(
        path_in_cp,
        labels="inferred",
        label_mode="categorical",
        class_names=[
            "discocyte",
            "holly_leaf",
            "granular",
            "sickle",
            "echinocyte",
        ],
        color_mode="grayscale",
        batch_size=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=93,
        validation_split=0.2,
        subset="both",
        data_format="channels_last",
        verbose=True,
    )

    train_ds_cp, test_ds_cp = [
        ds.map(
            lambda x, y: (tf.image.grayscale_to_rgb(x), y), num_parallel_calls=AUTOTUNE
        )
        for ds in (train_ds_cp, test_ds_cp)
    ]
    train_ds_cp = expand_ds(train_ds_cp)

    train_ds_cp, test_ds_cp = [
        ds.map(
            lambda x, y: (add_source_to_dataset(x, y, 1)), num_parallel_calls=AUTOTUNE
        )
        for ds in (train_ds_cp, test_ds_cp)
    ]

    train_ds = tf.data.Dataset.sample_from_datasets(
        [train_ds_im, train_ds_cp],
        weights=[0.5, 0.5],
    )

    test_ds = tf.data.Dataset.sample_from_datasets(
        [test_ds_im, test_ds_cp], weights=[0.5, 0.5]
    )

    train_ds = (
        train_ds.cache().batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    )
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    # Set up models to compare
    earlystopper = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=3,
        mode="min",
        restore_best_weights=True,
    )
    lr_scheduler = keras.callbacks.LearningRateScheduler(learning_rate_schedule)
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    EPOCHS = 200

    models_dict = {
        "EfficientNetB0": keras.applications.EfficientNetB0,
    }

    results = {}
    history_dict = {}

    for model_name, model_class in models_dict.items():
        print(f"Training {model_name}...")

        base_model = model_class(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(IMG_SIZE, IMG_SIZE, 4),
            pooling="None",
            classes=5,
            classifier_activation="softmax",
        )
        # print(base_model.summary())
        base_model.trainable = True

        model = Sequential()
        model.add(base_model)

        model.add(Dense(base_model.output_shape[-1], kernel_regularizer=L2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(
            Dense((base_model.output_shape[-1] // 2), kernel_regularizer=L2(0.01))
        )
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dense(124, kernel_regularizer=L2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(5, activation="softmax"))

        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        print(f"Compiled {model_name} model")

        with tf.device("GPU:0"):
            history = model.fit(
                train_ds,
                validation_data=test_ds,
                callbacks=[earlystopper, lr_scheduler],
                epochs=EPOCHS,
                verbose=1,
                validation_freq=1,
            )
            model.save(path_out / f"{model_name}.keras")
            y_test = np.concatenate([y.numpy() for _, y in test_ds])
            y_test = y_test.argmax(axis=1)
            y_pred = model.predict(test_ds)
            y_pred = y_pred.argmax(axis=1)

        print(f"y_test shape: {y_test.shape}")
        print(f"y_pred shape: {y_pred.shape}")
        accuracy = balanced_accuracy_score(y_test, y_pred)
        f1_score_model = f1_score(
            y_test, y_pred, average="weighted", labels=np.unique(y_pred)
        )
        precision = precision_score(
            y_test, y_pred, average="weighted", labels=np.unique(y_pred)
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", labels=np.unique(y_pred)
        )

        scores = {
            "test_balanced_accuracy": accuracy,
            "test_f1_weighted": f1_score_model,
            "test_precision_weighted": precision,
            "test_recall_weighted": recall,
        }

        results[model_name] = {"scores": scores}
        history_dict[model_name] = {"history": history.history}
        # Save history_dict using pickle
        with open(path_out / f"{model_name}_history.pkl", "wb") as f:
            pickle.dump(history.history, f)
        # get_confusion_matrix(model_name, y_test, y_pred, class_names, path_out)

    # Convert results to DataFrame
    results_df = pd.DataFrame({k: v["scores"] for k, v in results.items()}).T
    results_df.to_csv(path_out / "models_results.csv", index=True)

    for model_name, _ in models_dict.items():
        history = history_dict[model_name]["history"]
        plot_history(model_name, history, ["loss", "val_loss"], path_out)
        plot_history(model_name, history, ["accuracy", "val_accuracy"], path_out)


if __name__ == "__main__":
    main()
