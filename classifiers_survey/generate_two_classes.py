import numpy as np
from matplotlib import pyplot as plt

# data properties
n_samples = 1000
low_limit = -2
high_limit = 2
epsilon = 0.3

# decision_boundary = [constant, x1, x2, x1^2, x1x2, x2^2]
decision_boundary = [-1, 0, 0, 1, 0, 1]

X = np.random.uniform(low=low_limit, high=high_limit, size=(n_samples, 2))
Y = np.zeros(shape=(n_samples, 1), dtype=np.int32)

index_delete = []
for k in range(X.shape[0]):
    x = X[k, :]
    reg_result = np.dot(decision_boundary, np.array([1, x[0], x[1], x[0]*x[0], x[0]*x[1], x[1]*x[1]]))
    if abs(reg_result) < epsilon:
        index_delete.append(k)
    elif reg_result > 0:
        Y[k] = 1

X = np.delete(X, index_delete, axis=0)
Y = np.delete(Y, index_delete)

# balance the dataset
x_pos = X[Y == 1, :]
x_neg = X[Y == 0, :]

n_pos = x_pos.shape[0]
n_neg = x_neg.shape[0]

if n_pos > n_neg:
    n_class = n_neg
    x_pos = x_pos[:n_class, :]
else:
    n_class = n_pos
    x_neg = x_neg[:n_class, :]

x_pos = np.column_stack([x_pos, np.ones(shape=(n_class, 1), dtype=np.int32)])
x_neg = np.column_stack([x_neg, np.zeros(shape=(n_class, 1), dtype=np.int32)])

print('n_class = {:d}'.format(n_class))

data = np.row_stack([x_pos, x_neg])
np.random.shuffle(data)

print(data)

# =============
# plot the data
# =============
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

# =====================
# save the data to file
# =====================
# np.savez('two_classes.npz', data=data)



