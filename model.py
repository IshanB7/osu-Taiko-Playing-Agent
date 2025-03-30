import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os

class OsuTaikoGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_paths, labels, batch_size, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size

        self.image_paths = image_paths
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

            # img_array = np.round(img_array)
            batch_images.append(img_array)

        x = np.array(batch_images)
        y = self.labels[batch_indices]

        return x, y
    
    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        self.indices = np.random.permutation(self.indices)

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

def get_indices(dataset_filepath, num_train, oversample):
    dir_songs = sorted(os.listdir(dataset_filepath))
    
    if not dir_songs:
        exit("Dataset is empty")

    song_dirs = dir_songs[:len(dir_songs)//2]
    selected_songs = sorted(np.random.choice(song_dirs, num_train, replace=False))
    # selected_songs = ['1']

    all_labels = []
    all_image_paths = []

    for song_id in selected_songs:
        labels_path = os.path.join(dataset_filepath, f"labels_{song_id}.txt")
        images_path = os.path.join(dataset_filepath, song_id)

        labels = np.loadtxt(labels_path, dtype='str')

        image_paths = [os.path.join(images_path, f"image_{i}.png") for i in range(len(labels))]

        all_labels.append(labels)
        all_image_paths.extend(image_paths)

    all_image_paths = np.array(all_image_paths)
    all_labels = np.concatenate(all_labels)

    unique_labels, counts = np.unique(all_labels, return_counts=True)
    balanced_indices = []

    if oversample:
        max_count = max(counts)

        for label in unique_labels:
            class_indices = np.where(all_labels == label)[0]
            class_size = len(class_indices)

            repeat_factor = max_count // class_size
            for _ in range(repeat_factor):
                balanced_indices.extend(np.random.choice(class_indices, class_size, replace=False))

            balanced_indices.extend(np.random.choice(class_indices, max_count - (repeat_factor * class_size), replace=False))
            
    else:
        min_count = min(counts)

        for label in unique_labels:
            class_indices = np.where(all_labels == label)[0]
            balanced_indices.extend(np.random.choice(class_indices, min_count, replace=False))

    return all_image_paths[balanced_indices], all_labels[balanced_indices], selected_songs

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

def OsuTaikoModel(dataset_filepath, num_train):

    image_paths, labels, selected_songs = get_indices(dataset_filepath, num_train, oversample=False)

    batch_size = 32
    train_gen = OsuTaikoGenerator(image_paths, labels, batch_size)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(16, 16, 3)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.AvgPool2D((2, 2), padding='same'),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AvgPool2D((2, 2), padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=50, factor=0.2)
    ]

    model.fit(train_gen, epochs=1000, verbose=1, callbacks=callbacks)
    print(f"Trained on {selected_songs}")

    return model

model = OsuTaikoModel('./songs', 2)
model.save('osu_agent.keras')