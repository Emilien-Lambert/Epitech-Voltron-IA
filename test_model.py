import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import pathlib
import numpy as np


checkpoint_path = "training_1/cp.ckpt"
data_dir = pathlib.Path("test_data")

batch_size = 32
img_height = 216
img_width = 384


ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  image_size=(img_height, img_width))

# Define a simple sequential model
def create_model():
    num_classes = 2

    model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    print("Model created : \n")
    model.summary()

    return model

# Create a basic model instance
model = create_model()


# Evaluate the model
loss, acc = model.evaluate(ds, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


# Loads the weights
model.load_weights(checkpoint_path)


# Re-evaluate the model
loss, acc = model.evaluate(ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))