import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, layers, regularizers
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
)
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inceptionV3_preprocess_input,
)
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as resnet50v2_preprocess_input,
)
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenetv2_preprocess_input,
)
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnetb0_preprocess_input,
)

from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)


tf.random.set_seed(42)


def get_confusion_matrix(model_name, y_test, y_pred, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))  # Explicitly create figure
    cm = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        xticks_rotation="vertical",
        colorbar=False,
        normalize="true",
        display_labels=class_names,
    )

    plt.rc("font", size=12)
    ax.set_title(f"Confusion Matrix {model_name}")
    plt.savefig(Path.cwd().parent / "output" / f"confusion_matrix_{model_name}.png")
    plt.close(fig)


def plot_history(model_name, history, metrics):
    fig = plt.figure()  # Create a new figure before plotting
    sns.lineplot(data=history[metrics[0]], label=metrics[0])
    sns.lineplot(data=history[metrics[1]], label=metrics[1])
    plt.xlabel("epochs")
    plt.ylabel("metric")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        Path.cwd().parent / "output" / "{model_name}_{metrics}_history.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def learning_rate_schedule(epoch, lr):
    if epoch < 3:
        return (lr * tf.math.exp(0.5)).numpy()
    if epoch < 6:
        return lr
    else:
        return (lr * tf.math.exp(-0.1)).numpy()


def main():
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 200

    path_in = Path.cwd().parent / "resources/imagestream"

    train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
        path_in,
        labels="inferred",
        label_mode="categorical",
        class_names=["discocyte", "echinocyte", "granular", "holly_leaf", "sickle"],
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

    data_augmentation = Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ]
    )

    class_names = test_ds.class_names
    train_ds = train_ds.map(
        lambda x, y: (tf.image.grayscale_to_rgb(x), y), num_parallel_calls=AUTOTUNE
    )
    rgb_test_ds = test_ds.map(
        lambda x, y: (tf.image.grayscale_to_rgb(x), y), num_parallel_calls=AUTOTUNE
    )
    aug_train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )

    earlystopper = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, verbose=3, mode="min", restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    preprocess_input_dict = {
        "EfficientNetB0": efficientnetb0_preprocess_input,
        "InceptionV3": inceptionV3_preprocess_input,
        "ResNet50V2": resnet50v2_preprocess_input,
        "MobileNetV2": mobilenetv2_preprocess_input,
    }

    # "ResNet50V2": tf.keras.applications.ResNet50V2,
    # "MobileNetV2": tf.keras.applications.MobileNetV2,
    # "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    models_dict = {
        "InceptionV3": tf.keras.applications.InceptionV3,
    }
    results = {}
    history_dict = {}

    for model_name, model_class in models_dict.items():
        print(f"Training {model_name}...")

        base_model = model_class(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling="None",
            classes=6,
            classifier_activation="softmax",
        )
        base_model.trainable = True

        model = Sequential()
        model.add(base_model)
        model.add(
            Dense(base_model.output_shape[-1], kernel_regularizer=regularizers.L2(0.01))
        )
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(
            Dense(
                (base_model.output_shape[-1] // 2),
                kernel_regularizer=regularizers.L2(0.01),
            )
        )
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dense(124, kernel_regularizer=regularizers.L2(0.01)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(6, activation="softmax"))

        preprocess_input = preprocess_input_dict[model_name]

        train_ds = aug_train_ds.map(
            lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE
        )
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

        test_ds = rgb_test_ds.map(
            lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE
        )
        test_ds = test_ds.batch(BATCH_SIZE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        print(f"Preprocessed {model_name} data")

        with tf.device("GPU:0"):
            model.compile(
                optimizer=optimizer,
                loss=CategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"],
            )
            print(f"Compiled {model_name} model")

            history = model.fit(
                train_ds,
                validation_data=test_ds,
                callbacks=[earlystopper, lr_scheduler],
                epochs=EPOCHS,
                validation_freq=1,
                shuffle=True,
            )
            model.save(
                f"/home/t.afanasyeva/deep_learning_anaemias/output/250205_{model_name}.keras"
            )
        y_test = tf.concat([y for _, y in test_ds], axis=0)
        y_test = np.argmax(y_test, axis=1)
        y_pred = model.predict(test_ds)
        y_pred = y_pred.argmax(axis=1)

        accuracy = balanced_accuracy_score(y_test, y_pred)
        f1_score_model = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        scores = {
            "test_balanced_accuracy": accuracy,
            "test_f1_weighted": f1_score_model,
            "test_precision_weighted": precision,
            "test_recall_weighted": recall,
        }

        results[model_name] = {"scores": scores}
        history_dict[model_name] = {"history": history.history}

        # Save results and history using pickle
        with open(
            f"/home/t.afanasyeva/deep_learning_anaemias/output/{model_name}_score.pkl",
            "wb",
        ) as f:
            pickle.dump(results, f)
        with open(
            f"/home/t.afanasyeva/deep_learning_anaemias/output/{model_name}_history.pkl",
            "wb",
        ) as f:
            pickle.dump(history_dict, f)

        get_confusion_matrix(model_name, y_test, y_pred, class_names)

        # Convert results to DataFrame
        results_df = pd.DataFrame({k: v["scores"] for k, v in results.items()}).T
        results_df.to_csv(
            f"/home/t.afanasyeva/deep_learning_anaemias/output/250206_{model_name}_results.csv",
            index=True,
        )

    for model_name, _ in models_dict.items():
        history = history_dict[model_name]["history"]
        history["val_loss"] = [val for val in history["val_loss"] for _ in range(2)]
        history["val_accuracy"] = [
            val for val in history["val_accuracy"] for _ in range(2)
        ]
        plot_history(model_name, history, ["loss", "val_loss"])
        plot_history(model_name, history, ["accuracy", "val_accuracy"])


if __name__ == "__main__":
    main()
