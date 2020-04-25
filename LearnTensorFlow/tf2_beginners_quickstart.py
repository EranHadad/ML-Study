import tensorflow as tf

# Load and prepare the MNIST dataset.
mnist = tf.keras.datasets.mnist

# Convert the samples from integers to floating-point numbers:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)  # without activation function, the output of this layer would be the logits (log-odds)
])

# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions = model(x_train[:1]).numpy()
print(predictions)


print('initial probablities:', tf.nn.softmax(predictions).numpy())


# Choose an optimizer and loss function for training:
# ===================================================
# The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index
# and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print('initial loss:', loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# The Model.fit method adjusts the model parameters to minimize the loss:
model.fit(x_train, y_train, epochs=5, verbose=2)

# The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".
model.evaluate(x_test, y_test, verbose=2)

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
# train with 'model', predict with 'probability_model'
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]).numpy().round(2))

