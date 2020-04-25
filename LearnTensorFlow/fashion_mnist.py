# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from matplotlib import pyplot as plt

print(tf.__version__)

# Import the Fashion MNIST dataset
# ================================
# The Fashion MNIST dataset contains 70,000 grayscale images in 10 categories.
# The images show individual articles of clothing at low resolution (28 by 28 pixels).
# Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned
# to classify images

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# extract data for validation
num_samples_validation = 0.1 * train_images.shape[0]
num_samples_validation = tf.cast(num_samples_validation, tf.int64)

indices = np.arange(train_images.shape[0])
np.random.shuffle(indices)
train_images = train_images[indices]
train_labels = train_labels[indices]

validation_images = train_images[:num_samples_validation]
validation_labels = train_labels[:num_samples_validation]

train_images = train_images[num_samples_validation:]
train_labels = train_labels[num_samples_validation:]

print('training set: {ntrain:d} samples. test set: {ntest:d} samples'.format(ntrain=train_images.shape[0],
                                                                             ntest=test_images.shape[0]))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data
# ===================
# The data must be preprocessed before training the network. If you inspect the first image in the training set,
# you will see that the pixel values fall in the range of 0 to 255:
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''
train_images = train_images / 255.0
test_images = test_images / 255.0

# To verify that the data is in the correct format and that you're ready to build and train the network, let's
# display the first 25 images from the training set and display the class name below each image.
'''
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

# Build the model
# ===============
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile the model
# =================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
# ===============
early_stopping = keras.callbacks.EarlyStopping(patience=2)

# To do: validation set
model.fit(train_images, train_labels,
          validation_data=(validation_images, validation_labels), callbacks=[early_stopping],
          epochs=10, batch_size=200, verbose=2)

# Evaluate accuracy
# =================
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
# =================
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)


# Verify predictions
# ==================
def plot_image(ind, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[ind], img[ind]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(ind, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[ind]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Let's look at the 0th image, predictions, and prediction array. Correct prediction labels are blue and incorrect
# prediction labels are red. The number gives the percentage (out of 100) for the predicted label.
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

for i in range(test_labels.shape[0]):
    if np.argmax(predictions[i]) != test_labels[i]:
        ifalse = i
        print('first false prediction found at index {0:d}'.format(ifalse))
        break

i = ifalse
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Use the trained model
# =====================
# Finally, use the trained model to make a prediction about a single image.
# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once.
# Accordingly, even though you're using a single image, you need to add it to a list:
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

print(img.shape)

# Now predict the correct label for this image:
predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# keras.Model.predict returns a list of lists—one list for each image in the batch of data.
# Grab the predictions for our (only) image in the batch
print('predicted label:', np.argmax(predictions_single[0]))
