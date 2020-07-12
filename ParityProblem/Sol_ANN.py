import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import all_parity_pairs

# Data:
# =====
nbits = 3
x, y = all_parity_pairs(nbits)

inputs = x.astype(np.float)
targets = y.astype(np.int)

# Model:
# ======
# Set the input and output sizes
input_size = nbits
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 2**(nbits-1)

# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and
    # the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),  # 2nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax')  # output layer
])

# Choose the optimizer and the loss function:
# ===========================================
# we define the optimizer we'd like to use,
# the loss function,
# and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training:
# =========
# That's where we train the model we have built.

# set the batch size
batch_size = 20

# set a maximum number of training epochs
max_epochs = 100

# fit the model
# note that this time the train, validation and test data are not iterable
model.fit(inputs,  # train inputs
          targets,  # train targets
          batch_size=batch_size,  # batch size
          epochs=max_epochs,  # epochs that we will train for
          verbose=2  # making sure we get enough information about the training process
          )

"""
Description of the algorithm
    - hidden layer with 'relu' activation
    - output layer with 'softmax' activation
    - compile: 'adam' optimizer and 'cross-entropy' loss
    - fit: ephochs: 400, batch size: 20
    - targets are as type int (important for the sparse cross-entropy)
    - regularization
    - print each epoch, plot the cost function
    
To check:
    - shuffle the training data before each epoch
"""

