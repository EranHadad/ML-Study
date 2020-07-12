import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

DEBUG_PLOT = False

# load the data from file
info = np.load('two_classes.npz')
data = info['data']

# Feature Matrix
x = data[:, :-1]

# Data labels
y = data[:, -1]

print("Shape of Feature Matrix:", x.shape)
print("Shape Label Vector:", y.shape)


# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2)
])

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model:
model.fit(x, y, batch_size=5, epochs=10, verbose=2)

# Make predictions With the model trained, you can use it to make predictions about some images. The model's linear
# outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(x)
predictions = np.argmax(predictions, axis=1)

n_correct = (predictions == y).sum()
n_total = y.shape[0]

# Calculating the Decision Boundary
W, b = model.get_weights()
decision_boundary_x = np.linspace(start=np.min(x[:, 0]), stop=np.max(x[:, 0]), num=100)
decision_boundary_y = (decision_boundary_x * (W[0, 1] - W[0, 0]) + b[1] - b[0]) / (W[1, 0] - W[1, 1])  # solution #1

if DEBUG_PLOT:

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Positive Data Points
    x_pos = x[y == 1, :]

    # Negative Data Points
    x_neg = x[y == 0, :]

    # Plotting the Positive Data Points
    ax1.scatter(x_pos[:, 0], x_pos[:, 1], color='blue', label='Positive')

    # Plotting the Negative Data Points
    ax1.scatter(x_neg[:, 0], x_neg[:, 1], color='red', label='Negative')

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('True Classes')
    ax1.legend()

    # plot model's predictions
    # Positive Data Points
    x_pos = x[predictions == 1, :]

    # Negative Data Points
    x_neg = x[predictions == 0, :]

    # Plotting the Positive Data Points
    ax2.scatter(x_pos[:, 0], x_pos[:, 1], color='blue', label='Positive')

    # Plotting the Negative Data Points
    ax2.scatter(x_neg[:, 0], x_neg[:, 1], color='red', label='Negative')

    ax2.plot(decision_boundary_x, decision_boundary_y)

    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Predicted Classes (accuracy = {:d}/{:d} = {:.2f}%)'.format(n_correct,
                                                                              n_total,
                                                                              n_correct / n_total * 100))
    ax2.legend()
    ax2.set_ylim(ax1.get_ylim())

    plt.show()

# Plotting decision regions
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

Z = probability_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
# axarr[idx[0], idx[1]].set_title(tt)
plt.show()
