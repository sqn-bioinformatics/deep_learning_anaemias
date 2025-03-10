import os
import umap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from sklearn.manifold import TSNE
from pathlib import Path

tf.keras.backend.clear_session()
tf.config.set_visible_devices([], "GPU")


def create_encoder(representation_dim):
    encoder = tf.keras.Sequential(
        [
            tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, pooling="avg"
            ),
            layers.Dense(representation_dim),
        ]
    )
    return encoder


class RepresentationLearner(tf.keras.Model):
    def __init__(
        self,
        encoder,
        projection_units,
        num_augmentations,
        temperature=1.0,
        dropout_rate=0.1,
        l2_normalize=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        # Create projection head.
        self.projector = tf.keras.Sequential(
            [
                layers.Dropout(dropout_rate),
                layers.Dense(units=projection_units, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vectors, batch_size):
        num_augmentations = tf.keras.ops.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = tf.keras.utils.normalize(feature_vectors)
        # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
        logits = (
            tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
            / self.temperature
        )
        # Apply log-max trick for numerical stability.
        logits_max = tf.keras.ops.max(logits, axis=1)
        logits = logits - logits_max
        # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
        # targets is a matrix consits of num_augmentations submatrices of shape [batch_size * batch_size].
        # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
        targets = tf.keras.ops.tile(
            tf.eye(batch_size), [num_augmentations, num_augmentations]
        )
        # Compute cross entropy loss
        return tf.keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    def call(self, inputs):
        # Preprocess the input images.
        preprocessed = data_preprocessing(inputs)
        # Create augmented versions of the images.
        augmented = []
        for _ in range(self.num_augmentations):
            augmented.append(data_augmentation(preprocessed))
        augmented = layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(augmented)
        # Apply projection head.
        return self.projector(features)

    def train_step(self, inputs):
        batch_size = tf.keras.ops.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        batch_size = tf.keras.ops.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


class ClustersConsistencyLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, target, similarity, sample_weight=None):
        # Set targets to be ones.
        target = tf.keras.ops.ones_like(similarity)
        # Compute cross entropy loss.
        loss = tf.keras.losses.binary_crossentropy(
            y_true=target, y_pred=similarity, from_logits=True
        )
        return tf.keras.ops.mean(loss)


class ClustersEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, entropy_loss_weight=1.0):
        super().__init__()
        self.entropy_loss_weight = entropy_loss_weight

    def __call__(self, target, cluster_probabilities, sample_weight=None):
        # Ideal entropy = log(num_clusters).
        num_clusters = tf.keras.ops.cast(
            tf.keras.ops.shape(cluster_probabilities)[-1], "float32"
        )
        target = tf.keras.ops.log(num_clusters)
        # Compute the overall clusters distribution.
        cluster_probabilities = tf.keras.ops.mean(cluster_probabilities, axis=0)
        # Replacing zero probabilities - if any - with a very small value.
        cluster_probabilities = tf.keras.ops.clip(cluster_probabilities, 1e-8, 1.0)
        # Compute the entropy over the clusters.
        entropy = -tf.keras.ops.sum(
            cluster_probabilities * tf.keras.ops.log(cluster_probabilities)
        )
        # Compute the difference between the target and the actual.
        loss = target - entropy
        return loss


def create_clustering_model(encoder, num_clusters, name=None):
    inputs = tf.keras.Input(shape=input_shape)
    # Preprocess the input images.
    preprocessed = data_preprocessing(inputs)
    # Apply data augmentation to the images.
    augmented = data_augmentation(preprocessed)
    # Generate embedding representations of the images.
    features = encoder(augmented)
    # Assign the images to clusters.
    outputs = layers.Dense(units=num_clusters, activation="softmax")(features)
    # Create the model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def create_clustering_learner(clustering_model):
    anchor = tf.keras.Input(shape=input_shape, name="anchors")
    neighbours = tf.keras.Input(
        shape=tuple([k_neighbours]) + input_shape, name="neighbours"
    )
    # Changes neighbours shape to [batch_size * k_neighbours, width, height, channels]
    neighbours_reshaped = tf.keras.ops.reshape(neighbours, tuple([-1]) + input_shape)
    # anchor_clustering shape: [batch_size, num_clusters]
    anchor_clustering = clustering_model(anchor)
    # neighbours_clustering shape: [batch_size * k_neighbours, num_clusters]
    neighbours_clustering = clustering_model(neighbours_reshaped)
    # Convert neighbours_clustering shape to [batch_size, k_neighbours, num_clusters]
    neighbours_clustering = tf.keras.ops.reshape(
        neighbours_clustering,
        (-1, k_neighbours, tf.keras.ops.shape(neighbours_clustering)[-1]),
    )
    # similarity shape: [batch_size, 1, k_neighbours]
    similarity = tf.keras.ops.einsum(
        "bij,bkj->bik",
        tf.keras.ops.expand_dims(anchor_clustering, axis=1),
        neighbours_clustering,
    )
    # similarity shape:  [batch_size, k_neighbours]
    similarity = layers.Lambda(
        lambda x: tf.keras.ops.squeeze(x, axis=1), name="similarity"
    )(similarity)
    # Create the model.
    model = tf.keras.Model(
        inputs=[anchor, neighbours],
        outputs=[similarity, anchor_clustering],
        name="clustering_learner",
    )
    return model


# Define the path to the directory containing the images
data_dir = "/home/t.afanasyeva/deep_learning_anaemias/resources/cytpix/augmented"
dir_out_path = Path(
    "/home/t.afanasyeva/deep_learning_anaemias/output/250208_semantic/train_ds/"
)

X, y = [], []

# Set a fixed image size
IMG_SIZE = (64, 64)  # Change as needed

# Loop through each folder in the directory
for folder_name in tqdm(os.listdir(data_dir), desc="Folders"):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        image_names = os.listdir(folder_path)
        for image_name in tqdm(image_names, desc=f"Images in {folder_name}"):
            image_path = os.path.join(folder_path, image_name)
            try:
                im = Image.open(image_path).convert("RGB")  # Ensure RGB mode
                im = im.resize(IMG_SIZE)  # Resize to fixed shape
                X.append(np.array(im))  # Convert to numpy array
                y.append(folder_name)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

# Convert the lists to numpy arrays
x_data = np.array(X, dtype=np.uint8)  # Specify dtype to avoid memory issues
y_data = np.array(y)

print("x_data shape:", x_data.shape)  # Should be (num_samples, 224, 224, 3)
print("y_data shape:", y_data.shape)

print(f"Loaded {len(X)} images with {len(np.unique(y))} unique labels.")

classes = ["discocyte", "echinocyte", "granular", "holly_leaf", "sickle"]
epochs_repl = 200
epochs_clustering = 100
target_size = 64  # Resize the input images.
representation_dim = 512  # The dimensions of the features vector.
projection_units = 128  # The projection head of the representation learner.
num_clusters = 5  # Number of clusters.
k_neighbours = 5  # Number of neighbours to consider during cluster learning.
tune_encoder_during_clustering = False  # Freeze the encoder in the cluster learning.
input_shape = (64, 64, 3)  # The input shape of the images for the models.

data_preprocessing = tf.keras.Sequential(
    [
        layers.Resizing(target_size, target_size),
        layers.Normalization(),
    ]
)
# Compute the mean and the variance from the data for normalization.
data_preprocessing.layers[-1].adapt(x_data)
print("Data preprocessing completed.")


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomTranslation(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
        ),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.15, fill_mode="nearest"),
        layers.RandomZoom(
            height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
        ),
    ]
)
print("Data augmentation completed.")
# Create vision encoder.
encoder = create_encoder(representation_dim)
# Create representation learner.
representation_learner = RepresentationLearner(
    encoder, projection_units, num_augmentations=2, temperature=0.1
)
# Create a a Cosine decay learning rate scheduler.
lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=500, alpha=0.1
)
# Compile the model.
representation_learner.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr_scheduler, weight_decay=0.0001
    ),
    jit_compile=False,
)
# Fit the model.
history = representation_learner.fit(
    x=x_data,
    batch_size=512,
    epochs=epochs_repl,  # for better results, increase the number of epochs to 500.
)
# Save the representation learner model
with open(dir_out_path / "representation_learner.pkl", "wb") as f:
    pickle.dump(representation_learner, f)

plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(dir_out_path / "representation_learner_loss_plot.png", bbox_inches="tight")


"""
### Generate the embeddings for the images
"""
batch_size = 500  # Total number of images should be fully dividable by it
# Get the feature vector representations of the images
feature_vectors = encoder.predict(x_data, batch_size=batch_size, verbose=1)
# Normalize the feature vectores
feature_vectors = tf.keras.utils.normalize(feature_vectors)

# Save the feature vectors
with open(dir_out_path / "feature_vectors.pkl", "wb") as f:
    pickle.dump(feature_vectors, f)
print("Feature vectors shape:", feature_vectors.shape)


"""
### Find the *k* nearest neighbours for each embedding
"""

neighbours = []
num_batches = feature_vectors.shape[0] // batch_size
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    current_batch = feature_vectors[start_idx:end_idx]
    # Compute the dot similarity.
    similarities = tf.linalg.matmul(current_batch, feature_vectors, transpose_b=True)
    # Get the indices of most similar vectors.
    _, indices = tf.keras.ops.top_k(similarities, k=k_neighbours + 1, sorted=True)
    # Add the indices to the neighbours.
    neighbours.append(indices[..., 1:])


neighbours = np.reshape(np.array(neighbours), (-1, k_neighbours))

# Save the neighbours
with open(dir_out_path / "neighbours.pkl", "wb") as f:
    pickle.dump(neighbours, f)


"""
Let's display some neighbors on each row
"""
nrows = 4
ncols = k_neighbours + 1

plt.figure(figsize=(12, 12))
position = 1
for _ in range(nrows):
    anchor_idx = np.random.choice(range(x_data.shape[0]))
    neighbour_indicies = neighbours[anchor_idx]
    indices = [anchor_idx] + neighbour_indicies.tolist()
    for j in range(ncols):
        plt.subplot(nrows, ncols, position)
        plt.imshow(x_data[indices[j]].astype("uint8"))
        plt.axis("off")
        position += 1
plt.savefig(dir_out_path / "neighbours_plot.png", bbox_inches="tight")
plt.close()

# If tune_encoder_during_clustering is set to False,
# then freeze the encoder weights.
for layer in encoder.layers:
    layer.trainable = tune_encoder_during_clustering
# Create the clustering model and learner.
clustering_model = create_clustering_model(encoder, num_clusters, name="clustering")
clustering_learner = create_clustering_learner(clustering_model)
# Instantiate the model losses.
losses = [ClustersConsistencyLoss(), ClustersEntropyLoss(entropy_loss_weight=5)]
# Create the model inputs and labels.
inputs = {"anchors": x_data, "neighbours": tf.gather(x_data, neighbours)}
# labels = np.ones(shape=(x_data.shape[0]))

labels = [
    np.ones(shape=(x_data.shape[0], k_neighbours)),  # Corresponds to similarity
    np.ones(shape=(x_data.shape[0], num_clusters)),  # Corresponds to anchor_clustering
]
# labels[1] = tf.keras.utils.to_categorical(labels[1], num_classes=num_clusters)
# Compile the model.
clustering_learner.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001),
    loss=losses,
    jit_compile=False,
)
# Begin training the model.
clustering_learner.fit(x=inputs, y=labels, batch_size=512, epochs=epochs_clustering)

# Save the clustering learner model
with open(dir_out_path / "clustering_learner.pkl", "wb") as f:
    pickle.dump(clustering_learner, f)

plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(dir_out_path / "clustering_learner_loss_plot.png", bbox_inches="tight")
plt.show()
plt.close()

# Get the cluster probability distribution of the input images.
clustering_probs = clustering_model.predict(x_data, batch_size=batch_size, verbose=1)
# Get the cluster of the highest probability.
cluster_assignments = tf.keras.ops.argmax(clustering_probs, axis=-1).numpy()
# Store the clustering confidence.
# Images with the highest clustering confidence are considered the 'prototypes'
# of the clusters.
cluster_confidence = tf.keras.ops.max(clustering_probs, axis=-1).numpy()
clusters = defaultdict(list)
for idx, c in enumerate(cluster_assignments):
    clusters[c].append((idx, cluster_confidence[idx]))

non_empty_clusters = defaultdict(list)
for c in clusters.keys():
    if clusters[c]:
        non_empty_clusters[c] = clusters[c]

for c in range(num_clusters):
    print("cluster", c, ":", len(clusters[c]))
num_images = 5
plt.figure(figsize=(10, 10))
position = 1
for c in non_empty_clusters.keys():
    cluster_instances = sorted(
        non_empty_clusters[c], key=lambda kv: kv[1], reverse=True
    )

    for j in range(num_images):
        image_idx = cluster_instances[j][0]
        plt.subplot(len(non_empty_clusters), num_images, position)
        plt.imshow(x_data[image_idx].astype("uint8"))
        plt.axis("off")
        position += 1
plt.savefig(dir_out_path / "cluster_prototypes.png", bbox_inches="tight")
plt.close()

num_classes = len(classes)
cluster_label_counts = dict()

for c in range(num_clusters):
    cluster_label_counts[c] = [0] * num_classes
    instances = clusters[c]
    for i, _ in instances:
        label_index = classes.index(y_data[i])
        cluster_label_counts[c][label_index] += 1

    cluster_label_idx = np.argmax(cluster_label_counts[c])
    correct_count = np.max(cluster_label_counts[c])
    cluster_size = len(clusters[c])
    accuracy = (
        np.round((correct_count / cluster_size) * 100, 2) if cluster_size > 0 else 0
    )
    cluster_label = classes[cluster_label_idx]
    print("cluster", c, "label is:", cluster_label, " -  accuracy:", accuracy, "%")

# Reduce dimensions to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
feature_vectors_2d = tsne.fit_transform(feature_vectors)

# Assign cluster labels (these should be the predicted cluster assignments from the clustering model)
cluster_labels = clustering_model.predict(x_data).argmax(axis=1)

# Visualize using scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=feature_vectors_2d[:, 0],
    y=feature_vectors_2d[:, 1],
    hue=cluster_labels,
    palette=sns.color_palette("hsv", num_clusters),
    alpha=0.7,
)
plt.title("t-SNE Visualization of Clusters")
plt.savefig(dir_out_path / "tsne_clusters.png", bbox_inches="tight")
plt.close()


# Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
feature_vectors_2d = tsne.fit_transform(feature_vectors)

# Convert labels from (N,1) shape to (N,)
y_data_flat = y_data.flatten()

# Define a color palette
palette = sns.color_palette("hsv", num_classes)

# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=feature_vectors_2d[:, 0],
    y=feature_vectors_2d[:, 1],
    hue=y_data_flat,
    palette=palette,
    legend="full",
    alpha=0.7,
)

# Add title and legend
plt.title("t-SNE Visualization of Clusters with True Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig(dir_out_path / "tsne_clusters_labels.png", bbox_inches="tight")
plt.close()


# Reduce dimensions using UMAP
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
feature_vectors_2d = umap_reducer.fit_transform(feature_vectors)

# Plot clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=feature_vectors_2d[:, 0],
    y=feature_vectors_2d[:, 1],
    hue=cluster_labels,
    palette=sns.color_palette("hsv", num_clusters),
    alpha=0.7,
)
plt.title("UMAP Visualization of Clusters")
plt.savefig(dir_out_path / "umap_clusters.png", bbox_inches="tight")
plt.close()

# Reduce dimensions using UMAP
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.3)
feature_vectors_2d = umap_model.fit_transform(feature_vectors)

# Convert labels from (N,1) shape to (N,)
y_data_flat = y_data.flatten()

# Define a color palette
palette = sns.color_palette("hsv", num_classes)

# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=feature_vectors_2d[:, 0],
    y=feature_vectors_2d[:, 1],
    hue=y_data_flat,
    palette=palette,
    legend="full",
    alpha=0.7,
)

# Add title and legend
plt.title("UMAP Visualization of Clusters with True Labels")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.savefig(dir_out_path / "umap_clusters_labels.png", bbox_inches="tight")
plt.close()


# import os
# import umap
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import tensorflow as tf

# from tensorflow.keras import layers
# from tqdm import tqdm
# from collections import defaultdict
# from PIL import Image
# from sklearn.manifold import TSNE
# from pathlib import Path
# import pickle

# tf.keras.backend.clear_session()

# os.environ["KERAS_BACKEND"] = "tensorflow"

# # Define the path to the directory containing the images
# data_dir = "/home/t.afanasyeva/deep_learning_anaemias/resources/train"
# dir_out_path = Path(
#     "/home/t.afanasyeva/deep_learning_anaemias/output/250208_semantic/train_ds/"
# )

# # Initialize lists to hold the images and labels
# X = []
# y = []

# # Loop through each folder in the directory
# for folder_name in os.listdir(data_dir):
#     folder_path = os.path.join(data_dir, folder_name)
#     if os.path.isdir(folder_path):
#         image_names = os.listdir(folder_path)
#         for image_name in image_names:
#             image_path = os.path.join(folder_path, image_name)
#             im = Image.open(image_path).convert("RGB")
#             if im is not None:
#                 X.append(im)
#                 y.append(folder_name)

# # Convert the lists to numpy arrays
# x_data = np.array(X)
# y_data = np.array(y)

# print(f"Loaded {len(X)} images with {len(np.unique(y))} unique labels.")


# classes = ["discocyte", "echinocyte", "granular", "holly_leaf", "sickle"]
# target_size = 224  # Resize the input images.
# representation_dim = 512  # The dimensions of the features vector.
# projection_units = 128  # The projection head of the representation learner.
# num_clusters = 8  # Number of clusters.
# k_neighbours = 5  # Number of neighbours to consider during cluster learning.
# tune_encoder_during_clustering = False  # Freeze the encoder in the cluster learning.

# data_preprocessing = tf.keras.Sequential(
#     [
#         layers.Resizing(target_size, target_size),
#         layers.Normalization(),
#     ]
# )
# # Compute the mean and the variance from the data for normalization.
# data_preprocessing.layers[-1].adapt(x_data)

# data_augmentation = tf.keras.Sequential(
#     [
#         layers.RandomTranslation(
#             height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
#         ),
#         layers.RandomFlip(mode="horizontal"),
#         layers.RandomRotation(factor=0.15, fill_mode="nearest"),
#         layers.RandomZoom(
#             height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
#         ),
#     ]
# )


# def create_encoder(representation_dim):
#     encoder = tf.keras.Sequential(
#         [
#             tf.keras.applications.EfficientNetB0(
#                 include_top=False, weights=None, pooling="avg"
#             ),
#             layers.Dense(representation_dim),
#         ]
#     )
#     return encoder


# class RepresentationLearner(tf.keras.Model):
#     def __init__(
#         self,
#         encoder,
#         projection_units,
#         num_augmentations,
#         temperature=1.0,
#         dropout_rate=0.1,
#         l2_normalize=False,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.encoder = encoder
#         # Create projection head.
#         self.projector = tf.keras.Sequential(
#             [
#                 layers.Dropout(dropout_rate),
#                 layers.Dense(units=projection_units, use_bias=False),
#                 layers.BatchNormalization(),
#                 layers.ReLU(),
#             ]
#         )
#         self.num_augmentations = num_augmentations
#         self.temperature = temperature
#         self.l2_normalize = l2_normalize
#         self.loss_tracker = tf.keras.metrics.Mean(name="loss")

#     @property
#     def metrics(self):
#         return [self.loss_tracker]

#     def compute_contrastive_loss(self, feature_vectors, batch_size):
#         num_augmentations = tf.keras.ops.shape(feature_vectors)[0] // batch_size
#         if self.l2_normalize:
#             feature_vectors = tf.keras.utils.normalize(feature_vectors)
#         # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
#         logits = (
#             tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
#             / self.temperature
#         )
#         # Apply log-max trick for numerical stability.
#         logits_max = tf.keras.ops.max(logits, axis=1)
#         logits = logits - logits_max
#         # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
#         # targets is a matrix consits of num_augmentations submatrices of shape [batch_size * batch_size].
#         # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
#         targets = tf.keras.ops.tile(
#             tf.eye(batch_size), [num_augmentations, num_augmentations]
#         )
#         # Compute cross entropy loss
#         return tf.keras.losses.categorical_crossentropy(
#             y_true=targets, y_pred=logits, from_logits=True
#         )

#     def call(self, inputs):
#         # Preprocess the input images.
#         preprocessed = data_preprocessing(inputs)
#         # Create augmented versions of the images.
#         augmented = []
#         for _ in range(self.num_augmentations):
#             augmented.append(data_augmentation(preprocessed))
#         augmented = layers.Concatenate(axis=0)(augmented)
#         # Generate embedding representations of the images.
#         features = self.encoder(augmented)
#         # Apply projection head.
#         return self.projector(features)

#     def train_step(self, inputs):
#         batch_size = tf.keras.ops.shape(inputs)[0]
#         # Run the forward pass and compute the contrastive loss
#         with tf.GradientTape() as tape:
#             feature_vectors = self(inputs, training=True)
#             loss = self.compute_contrastive_loss(feature_vectors, batch_size)
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update loss tracker metric
#         self.loss_tracker.update_state(loss)
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}

#     def test_step(self, inputs):
#         batch_size = tf.keras.ops.shape(inputs)[0]
#         feature_vectors = self(inputs, training=False)
#         loss = self.compute_contrastive_loss(feature_vectors, batch_size)
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}


# # Create vision encoder.
# encoder = create_encoder(representation_dim)
# # Create representation learner.
# representation_learner = RepresentationLearner(
#     encoder, projection_units, num_augmentations=2, temperature=0.1
# )
# # Create a a Cosine decay learning rate scheduler.
# lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=0.001, decay_steps=500, alpha=0.1
# )
# # Compile the model.
# representation_learner.compile(
#     optimizer=tf.keras.optimizers.AdamW(
#         learning_rate=lr_scheduler, weight_decay=0.0001
#     ),
#     jit_compile=False,
# )
# # Fit the model.
# history = representation_learner.fit(
#     x=x_data,
#     batch_size=512,
#     epochs=200,  # for better results, increase the number of epochs to 500.
# )
# # Save the representation learner model
# with open(dir_out_path / "representation_learner.pkl", "wb") as f:
#     pickle.dump(representation_learner, f)

# plt.plot(history.history["loss"])
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.savefig(dir_out_path / "representation_learner_loss_plot.png")

# batch_size = 500
# # Get the feature vector representations of the images.
# feature_vectors = encoder.predict(x_data, batch_size=batch_size, verbose=1)
# # Normalize the feature vectores.
# feature_vectors = tf.keras.utils.normalize(feature_vectors)
# # Save the feature vectors
# with open(dir_out_path / "feature_vectors.pkl", "wb") as f:
#     pickle.dump(feature_vectors, f)

# neighbours = []
# num_batches = feature_vectors.shape[0] // batch_size
# for batch_idx in tqdm(range(num_batches)):
#     start_idx = batch_idx * batch_size
#     end_idx = start_idx + batch_size
#     current_batch = feature_vectors[start_idx:end_idx]
#     # Compute the dot similarity.
#     similarities = tf.linalg.matmul(current_batch, feature_vectors, transpose_b=True)
#     # Get the indices of most similar vectors.
#     _, indices = tf.keras.ops.top_k(similarities, k=k_neighbours + 1, sorted=True)
#     # Add the indices to the neighbours.
#     neighbours.append(indices[..., 1:])

#     # Save the neighbours
#     with open(dir_out_path / "neighbours.pkl", "wb") as f:
#         pickle.dump(neighbours, f)

# neighbours = np.reshape(np.array(neighbours), (-1, k_neighbours))
# nrows = 4
# ncols = k_neighbours + 1

# plt.figure(figsize=(12, 12))
# position = 1
# for _ in range(nrows):
#     anchor_idx = np.random.choice(range(x_data.shape[0]))
#     neighbour_indicies = neighbours[anchor_idx]
#     indices = [anchor_idx] + neighbour_indicies.tolist()
#     for j in range(ncols):
#         plt.subplot(nrows, ncols, position)
#         plt.imshow(x_data[indices[j]].astype("uint8"))
#         plt.title(classes[y_data[indices[j]]])
#         plt.axis("off")
#         position += 1
# plt.savefig(dir_out_path / "neighbours_plot.png")
# plt.show()
# plt.close()


# class ClustersConsistencyLoss(tf.keras.losses.Loss):
#     def __init__(self):
#         super().__init__()

#     def __call__(self, target, similarity, sample_weight=None):
#         # Set targets to be ones.
#         target = tf.keras.ops.ones_like(similarity)
#         # Compute cross entropy loss.
#         loss = tf.keras.losses.binary_crossentropy(
#             y_true=target, y_pred=similarity, from_logits=True
#         )
#         return tf.keras.ops.mean(loss)


# class ClustersEntropyLoss(tf.keras.losses.Loss):
#     def __init__(self, entropy_loss_weight=1.0):
#         super().__init__()
#         self.entropy_loss_weight = entropy_loss_weight

#     def __call__(self, target, cluster_probabilities, sample_weight=None):
#         # Ideal entropy = log(num_clusters).
#         num_clusters = tf.keras.ops.cast(
#             tf.keras.ops.shape(cluster_probabilities)[-1], "float32"
#         )
#         target = tf.keras.ops.log(num_clusters)
#         # Compute the overall clusters distribution.
#         cluster_probabilities = tf.keras.ops.mean(cluster_probabilities, axis=0)
#         # Replacing zero probabilities - if any - with a very small value.
#         cluster_probabilities = tf.keras.ops.clip(cluster_probabilities, 1e-8, 1.0)
#         # Compute the entropy over the clusters.
#         entropy = -tf.keras.ops.sum(
#             cluster_probabilities * tf.keras.ops.log(cluster_probabilities)
#         )
#         # Compute the difference between the target and the actual.
#         loss = target - entropy
#         return loss


# def create_clustering_model(encoder, num_clusters, name=None):
#     inputs = tf.keras.Input(shape=input_shape)
#     # Preprocess the input images.
#     preprocessed = data_preprocessing(inputs)
#     # Apply data augmentation to the images.
#     augmented = data_augmentation(preprocessed)
#     # Generate embedding representations of the images.
#     features = encoder(augmented)
#     # Assign the images to clusters.
#     outputs = layers.Dense(units=num_clusters, activation="softmax")(features)
#     # Create the model.
#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
#     return model


# input_shape = (224, 224, 3)


# def create_clustering_learner(clustering_model):
#     anchor = tf.keras.Input(shape=input_shape, name="anchors")
#     neighbours = tf.keras.Input(
#         shape=tuple([k_neighbours]) + input_shape, name="neighbours"
#     )
#     # Changes neighbours shape to [batch_size * k_neighbours, width, height, channels]
#     neighbours_reshaped = tf.keras.ops.reshape(neighbours, tuple([-1]) + input_shape)
#     # anchor_clustering shape: [batch_size, num_clusters]
#     anchor_clustering = clustering_model(anchor)
#     # neighbours_clustering shape: [batch_size * k_neighbours, num_clusters]
#     neighbours_clustering = clustering_model(neighbours_reshaped)
#     # Convert neighbours_clustering shape to [batch_size, k_neighbours, num_clusters]
#     neighbours_clustering = tf.keras.ops.reshape(
#         neighbours_clustering,
#         (-1, k_neighbours, tf.keras.ops.shape(neighbours_clustering)[-1]),
#     )
#     # similarity shape: [batch_size, 1, k_neighbours]
#     similarity = tf.keras.ops.einsum(
#         "bij,bkj->bik",
#         tf.keras.ops.expand_dims(anchor_clustering, axis=1),
#         neighbours_clustering,
#     )
#     # similarity shape:  [batch_size, k_neighbours]
#     similarity = layers.Lambda(
#         lambda x: tf.keras.ops.squeeze(x, axis=1), name="similarity"
#     )(similarity)
#     # Create the model.
#     model = tf.keras.Model(
#         inputs=[anchor, neighbours],
#         outputs=[similarity, anchor_clustering],
#         name="clustering_learner",
#     )
#     return model


# # If tune_encoder_during_clustering is set to False,
# # then freeze the encoder weights.
# for layer in encoder.layers:
#     layer.trainable = tune_encoder_during_clustering
# # Create the clustering model and learner.
# clustering_model = create_clustering_model(encoder, num_clusters, name="clustering")
# clustering_learner = create_clustering_learner(clustering_model)
# # Instantiate the model losses.
# losses = [ClustersConsistencyLoss(), ClustersEntropyLoss(entropy_loss_weight=5)]
# # Create the model inputs and labels.
# inputs = {"anchors": x_data, "neighbours": tf.gather(x_data, neighbours)}
# # labels = np.ones(shape=(x_data.shape[0]))

# labels = [
#     np.ones(shape=(x_data.shape[0], k_neighbours)),  # Corresponds to similarity
#     np.ones(shape=(x_data.shape[0], num_clusters)),  # Corresponds to anchor_clustering
# ]
# # labels[1] = tf.keras.utils.to_categorical(labels[1], num_classes=num_clusters)
# # Compile the model.
# clustering_learner.compile(
#     optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001),
#     loss=losses,
#     jit_compile=False,
# )
# # Begin training the model.
# clustering_learner.fit(x=inputs, y=labels, batch_size=512, epochs=100)
# # Save the clustering learner model
# with open(dir_out_path / "clustering_learner.pkl", "wb") as f:
#     pickle.dump(clustering_learner, f)

# plt.plot(history.history["loss"])
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.savefig(dir_out_path / "clustering_learner_loss_plot.png")
# plt.show()
# plt.close()

# # Get the cluster probability distribution of the input images.
# clustering_probs = clustering_model.predict(x_data, batch_size=batch_size, verbose=1)
# # Get the cluster of the highest probability.
# cluster_assignments = tf.keras.ops.argmax(clustering_probs, axis=-1).numpy()
# # Store the clustering confidence.
# # Images with the highest clustering confidence are considered the 'prototypes'
# # of the clusters.
# cluster_confidence = tf.keras.ops.max(clustering_probs, axis=-1).numpy()
# clusters = defaultdict(list)
# for idx, c in enumerate(cluster_assignments):
#     clusters[c].append((idx, cluster_confidence[idx]))

# non_empty_clusters = defaultdict(list)
# for c in clusters.keys():
#     if clusters[c]:
#         non_empty_clusters[c] = clusters[c]

# for c in range(num_clusters):
#     print("cluster", c, ":", len(clusters[c]))
# num_images = 5
# plt.figure(figsize=(10, 10))
# position = 1
# for c in non_empty_clusters.keys():
#     cluster_instances = sorted(
#         non_empty_clusters[c], key=lambda kv: kv[1], reverse=True
#     )

#     for j in range(num_images):
#         image_idx = cluster_instances[j][0]
#         plt.subplot(len(non_empty_clusters), num_images, position)
#         plt.imshow(x_data[image_idx].astype("uint8"))
#         plt.title(y_data[image_idx])
#         plt.axis("off")
#         position += 1
# plt.savefig(dir_out_path / "cluster_prototypes.png")
# plt.close()


# num_classes = len(np.unique(y_data))
# cluster_label_counts = dict()

# for c in range(num_clusters):
#     cluster_label_counts[c] = [0] * num_classes
#     instances = clusters[c]
#     for i, _ in instances:
#         label_index = classes.index(y_data[i])
#         cluster_label_counts[c][label_index] += 1

#     cluster_label_idx = np.argmax(cluster_label_counts[c])
#     correct_count = np.max(cluster_label_counts[c])
#     cluster_size = len(clusters[c])
#     accuracy = (
#         np.round((correct_count / cluster_size) * 100, 2) if cluster_size > 0 else 0
#     )
#     cluster_label = classes[cluster_label_idx]
#     print("cluster", c, "label is:", cluster_label, " -  accuracy:", accuracy, "%")

# # Reduce dimensions to 2D using t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# feature_vectors_2d = tsne.fit_transform(feature_vectors)

# # Assign cluster labels (these should be the predicted cluster assignments from the clustering model)
# cluster_labels = clustering_model.predict(x_data).argmax(axis=1)

# # Visualize using scatter plot
# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x=feature_vectors_2d[:, 0],
#     y=feature_vectors_2d[:, 1],
#     hue=cluster_labels,
#     palette=sns.color_palette("hsv", num_clusters),
#     alpha=0.7,
# )
# plt.title("t-SNE Visualization of Clusters")
# plt.show()
# plt.savefig(dir_out_path / "tsne_clusters.png")
# plt.close()


# # Reduce dimensions using t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# feature_vectors_2d = tsne.fit_transform(feature_vectors)

# # Convert labels from (N,1) shape to (N,)
# y_data_flat = y_data.flatten()

# # Define a color palette
# palette = sns.color_palette("hsv", num_classes)

# # Plot the clusters
# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x=feature_vectors_2d[:, 0],
#     y=feature_vectors_2d[:, 1],
#     hue=y_data_flat,
#     palette=palette,
#     legend="full",
#     alpha=0.7,
# )

# # Add title and legend
# plt.title("t-SNE Visualization of Clusters with True Labels")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.savefig(dir_out_path / "tsne_clusters_labels.png")
# plt.close()


# # Reduce dimensions using UMAP
# umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# feature_vectors_2d = umap_reducer.fit_transform(feature_vectors)

# # Plot clusters
# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x=feature_vectors_2d[:, 0],
#     y=feature_vectors_2d[:, 1],
#     hue=cluster_labels,
#     palette=sns.color_palette("hsv", num_clusters),
#     alpha=0.7,
# )
# plt.title("UMAP Visualization of Clusters")
# plt.savefig(dir_out_path / "umap_clusters.png")
# plt.close()

# # Reduce dimensions using UMAP
# umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
# feature_vectors_2d = umap_model.fit_transform(feature_vectors)

# # Convert labels from (N,1) shape to (N,)
# y_data_flat = y_data.flatten()

# # Define a color palette
# palette = sns.color_palette("hsv", num_classes)

# # Plot the clusters
# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     x=feature_vectors_2d[:, 0],
#     y=feature_vectors_2d[:, 1],
#     hue=y_data_flat,
#     palette=palette,
#     legend="full",
#     alpha=0.7,
# )

# # Add title and legend
# plt.title("UMAP Visualization of Clusters with True Labels")
# plt.xlabel("UMAP Component 1")
# plt.ylabel("UMAP Component 2")
# plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.savefig(dir_out_path / "umap_clusters_labels.png")
# plt.close()
