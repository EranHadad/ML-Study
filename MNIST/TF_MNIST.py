"""
MNIST - handwritten digit recognition
The dataset provides 70,000 images (28x28 pixels) of handwritten digits (1 digit per image).
The goal is to write an algorithm that detects which digit is written.
Since there are only 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), this is a classification problem with 10 classes.
"""

# ============================
# Import the relevant packages
# ============================
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# =========================
# Data: load and preprocess
# =========================

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label


scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

# =====
# Model
# =====

# Outline the model
# -----------------
input_size = 784
output_size = 10
hidden_layer_size = [100, 50]

layers_list = []
input_layer = tf.keras.layers.Flatten(input_shape=(28, 28, 1))  # input layer
layers_list.append(input_layer)
for num_units in hidden_layer_size:
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    hidden_layer = tf.keras.layers.Dense(units=num_units, activation='relu')  # 'relu', 'sigmoid'
    layers_list.append(hidden_layer)  # stack hidden layers
# the final layer is no different, we just make sure to activate it with softmax
output_layer = tf.keras.layers.Dense(units=output_size, activation='softmax')
layers_list.append(output_layer)

model = tf.keras.models.Sequential(layers_list)

# Choose the optimizer and loss function
# --------------------------------------
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ========
# Training
# ========

NUM_EPOCHS = 15
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, mode='min')

model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), validation_steps=1,
          verbose=2, callbacks=[early_stopping])

# ==============
# Test the model
# ==============

test_loss, test_accuracy = model.evaluate(test_data, steps=1)
# print('Test loss: {loss: .2f}. Test accuracy: {accuracy: .2f}'.format(loss=test_loss, accuracy=test_accuracy*100))
