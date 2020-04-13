import numpy as np
import mytensorflow as mytf
import matplotlib.pyplot as plt

# =========================
# define inputs and outputs
# =========================

nsamples = 1000

x1 = np.random.uniform(low=-10, high=10, size=(nsamples, 1))  # temp
x2 = np.random.uniform(low=-10, high=10, size=(nsamples, 1))  # temp
inputs = np.column_stack([x1, x2])  # temp

y1 = 2 * x1 - 3 * x2 + 5
y2 = np.zeros([nsamples, 1])
targets = np.column_stack([y1, y2])

# ===================
# construct the model
# ===================
model = mytf.Model([
    mytf.Layer(noutputs=2, init_range=0.1)
    # mytf.Layer(noutputs=3, init_range=0.1),
    # mytf.Layer(noutputs=2, init_range=0.1)
])

# ===============
# train the model
# ===============
model.fit(x=inputs, y=targets, batch_size=100)

print(model.layers[0].weights)
print(model.layers[-1].outputs[:10, :])

# ===============
# make prediction
# ===============
outputs = model.predict(inputs)

# We print the outputs and the targets in order to see if they have a linear relationship.
plt.scatter(outputs[:, 0], y1)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()

'''
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
'''