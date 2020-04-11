import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# print('TensorFlow version:', tf.__version__)

# ===============
# Data generation
# ===============
nsamples = 1000

xs = np.random.uniform(low=-10, high=10, size=(nsamples, 1))
zs = np.random.uniform(low=-10, high=10, size=(nsamples, 1))
generated_inputs = np.column_stack((xs, zs))

noise = np.random.uniform(low=-1, high=1, size=(nsamples, 1))

generated_targets = 2*xs - 3*zs + 5 + noise

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

# =======================
# Solving with TensorFlow
# =======================
training_data = np.load('TF_intro.npz')

input_size = 2
output_size = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size,
                          kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                          bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
])

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(optimizer=custom_optimizer, loss='huber_loss')

model.fit(x=training_data['inputs'], y=training_data['targets'], epochs=15, verbose=2)

# =============================
# Extract the weight and biases
# =============================
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
print(weights, biases)

# ======================================
# Extract the outputs (make predictions)
# ======================================
outputs = model.predict_on_batch(training_data['inputs'])

# =================
# Plotting the data
# =================
plt.plot(outputs, training_data['targets'])
plt.xlabel('outputs', fontsize=20)
plt.ylabel('targets', fontsize=20)
plt.grid()
plt.show()
