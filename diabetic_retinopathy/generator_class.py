"""
Adapted from:
A. Amidi, S. Amidi, "A detailed example of how to use data generators with Keras", n.d.,
available: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
Date of retrieval: 26 Jan 2023
"""


import numpy as np
import keras

from PIL import Image


# Generates data for keras
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels=1, n_classes=2, shuffle=True, directory=None):
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = labels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.directory = directory
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data using list index syntax
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_batch)

        return X, y

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_batch):
        # Generates data containing batch_size samples
        X = np.empty(shape=(self.batch_size, *self.dim, self.n_channels))
        y = np.empty(shape=(self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            # img = np.asarray(Image.open(self.directory + ID).convert('L')) / 255  # grayscale
            img = np.asarray(Image.open(self.directory + ID).resize(self.dim)) / 255
            X[i] = np.reshape(img, (*self.dim, self.n_channels))
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

