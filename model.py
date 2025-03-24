import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

class OsuTaikoGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_filepath, labels_filepath, use_rows, batch_size, **kwargs):
        super().__init__(**kwargs)

        self.dataset_filepath = dataset_filepath
        self.batch_size = batch_size

        self.labels = np.loadtxt('labels.txt', dtype='str')

        labels = []
        image_paths = []

        for i in use_rows:
            image_paths.append(os.path.join(dataset_filepath, f"image_{i}.png"))
            labels.append(self.labels[i])

        self.image_paths = np.array(image_paths)

        num_labels = LabelEncoder().fit_transform(labels)
        self.labels = tf.keras.utils.to_categorical(num_labels, num_classes=3)

        self.indices = np.arange(len(self.labels))
        self.shuffle()

    def __len__(self):
        return len(self.labels) // self.batch_size
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_paths = self.image_paths[batch_indices]

        batch_images = []
        for path in batch_paths:
            img = load_img(path, color_mode='rgb')
            img_array = img_to_array(img) / 255.0
            batch_images.append(img_array)

        x = np.array(batch_images)
        y = self.labels[batch_indices]

        return x, y
    
    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        self.indices = np.random.permutation(self.indices)

def OsuTaikoModel(dataset_filepath, labels_filepath, balance=True):

    all_idx = np.arange(len(os.listdir(dataset_filepath)))

    labels = np.loadtxt('labels.txt', dtype='str')

    if balance:
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = max(counts)
        balanced_indices = []

        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            class_size = len(class_indices)

            repeat_factor = max_count // class_size
            for _ in range(repeat_factor):
                balanced_indices.extend(np.random.choice(class_indices, class_size, replace=False))

            balanced_indices.extend(np.random.choice(class_indices, max_count - (repeat_factor * class_size), replace=False))
            all_idx = np.array(balanced_indices)

    train_gen = OsuTaikoGenerator(dataset_filepath, labels_filepath, all_idx, 32)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(16, 16, 3)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=20, verbose=1)

    return model

model = OsuTaikoModel('./images', './labels.txt')
model.save('osu_agent.keras')