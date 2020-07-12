import numpy as np
from matplotlib import pyplot as plt

N = 200  # number of examples drawn from each class

# generate features
class_0 = np.random.normal(size=(N, 2), loc=(2, 4), scale=1)
class_1 = np.random.normal(size=(N, 2), loc=(7, 8), scale=1)

# adding class labels {0, 1}
class_0 = np.column_stack([class_0, np.zeros(shape=(N, 1))])
class_1 = np.column_stack([class_1, np.ones(shape=(N, 1))])

# mix both classes to one dataset
data = np.row_stack([class_0, class_1])
np.random.shuffle(data)

# plot the data
trues = data[data[:, -1] == 1, :]
falses = data[data[:, -1] == 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(trues[:, 0], trues[:, 1], c='b', label='Trues')
ax.scatter(falses[:, 0], falses[:, 1], c='r', label='Falses')
plt.legend(loc='upper left')
ax.set_title("dataset")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.grid()
plt.show()

# save the data to file
np.savez('two_classes.npz', data=data)
