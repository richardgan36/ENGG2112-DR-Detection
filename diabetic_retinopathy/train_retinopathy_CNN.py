import tensorflow as tf
import numpy as np
import glob
import os

from keras import layers, Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from plot_metrics import plot_graph
from generator_class import DataGenerator

FILEPATH = '/Users/richardgan/Pictures/Machine Learning/diabetic_retinopathy/'
folders = ["Healthy", "Mild DR", "Moderate DR", "Proliferate DR", "Severe DR"]
stage_to_numeric = {"Healthy": 0, "Mild DR": 1, "Moderate DR": 2, "Proliferate DR": 3, "Severe DR": 4}
MODEL_NAME = 'model_3'


# Hyperparameters
filters_conv0 = 128
filters_conv1 = 64
dropout = 0.5
n_dense0 = 32
validation_split = 0.2
epochs = 30

class_params = {'dim': (64, 64),
                'batch_size': 64,
                'n_classes': 5,
                'n_channels': 3,
                'shuffle': True,
                'directory': FILEPATH}
                # directory is the name of the filepath where the training images are located


def main():
    partition = {'train': [], 'validation': []}
    labels = {}

    # For each folder, separate images into training and validation data
    for folder in folders:
        file_count = sum(1 for name in os.listdir(FILEPATH + folder))
        num_training_samples = np.ceil((1 - validation_split) * file_count)
        for i, filename in enumerate(glob.iglob(FILEPATH + folder + '/*')):
            ID = filename[len(FILEPATH):]  # Name of folder and image e.g. Healthy/Healthy_72.png
            if i < num_training_samples:
                partition['train'].append(ID)
            else:
                partition['validation'].append(ID)

            labels[ID] = stage_to_numeric[folder]  # assign a numeric label to the image ID

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **class_params)
    validation_generator = DataGenerator(partition['validation'], labels, **class_params)

    model = Sequential()
    model.add(Input(shape=(*class_params['dim'], class_params['n_channels'])))
    model.add(layers.Conv2D(filters_conv0, 3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(filters_conv1, 3, activation='relu'))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(n_dense0, activation='relu'))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(5, activation='softmax'))  # probability distribution output layer

    # model.summary()
    plot_model(model, 'trained_models/' + MODEL_NAME + '.png', show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   verbose=1)

    model_checkpoint = ModelCheckpoint(filepath='trained_models/' + MODEL_NAME + '.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True)

    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1,
                        use_multiprocessing=True,
                        workers=6)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epoch_range = np.arange(len(acc))
    plot_graph(epoch_range, acc, val_acc, loss, val_loss)


if __name__ == "__main__":
    main()
