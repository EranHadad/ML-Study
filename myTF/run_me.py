import numpy as np
import mytensorflow as mytf

inputs = np.random.uniform(low=-10, high=10, size=(1000, 2))  # temp
targets = np.random.uniform(low=-10, high=10, size=(1000, 2))  # temp

model = mytf.Model([
    mytf.Layer(noutputs=3, init_range=0.1),
    mytf.Layer(noutputs=2, init_range=0.1)
])

model.fit(x=inputs, y=targets, batch_size=100)

print(model.layers[0].weights)

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