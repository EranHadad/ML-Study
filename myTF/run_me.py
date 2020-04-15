import numpy as np
import mytensorflow as mytf
import matplotlib.pyplot as plt

# =========================
# define inputs and outputs
# =========================

nsamples = 1000
input_range = 10
noise_range = 1

x1 = np.random.uniform(low=-input_range, high=input_range, size=(nsamples, 1))
x2 = np.random.uniform(low=-input_range, high=input_range, size=(nsamples, 1))

generated_inputs = np.column_stack([x1, x2])

noise = np.random.uniform(low=-noise_range, high=noise_range, size=(nsamples, 1))

generated_targets = 2 * x1 - 3 * x2 + 5 + noise

np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

# ===================
# construct the model
# ===================
model = mytf.Model([
    mytf.Layer(noutputs=1, init_range=0.1)  # activation='logistic'
])

# ===============
# train the model
# ===============
training_data = np.load('TF_intro.npz')

model.fit(x=training_data['inputs'], y=training_data['targets'], epochs=50, batch_size=100,
          learning_rate=0.02, validation_split=0.2, early_stopping=True)

# print(model.layers[0].weights)
# print(model.layers[-1].outputs[:10, :])

# ===============
# make prediction
# ===============
outputs = model.predict(training_data['inputs'])

# We print the outputs and the targets in order to see if they have a linear relationship.
plt.scatter(outputs, training_data['targets'])
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()

# TO DO: compare to tensor flow output

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