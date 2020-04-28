import numpy as np
import theano
from theano import tensor as T
from matplotlib import pyplot as plt

from util import get_normalized_data, y2indicator


def glorot_uniform_initializer(ninputs, noutputs):
    x = np.sqrt(6 / (ninputs + noutputs))
    weights = np.random.uniform(low=-x, high=x, size=(ninputs, noutputs))
    return weights


# get the MNIST data
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

# apply one-hot encoding to the target vectors
Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
Ytest_ind = y2indicator(Ytest).astype(np.float32)

# initialize the model parameters
N, D = Xtrain.shape
batch_size = 500
num_batches = N // batch_size

M = 300
K = 10
w1_init = glorot_uniform_initializer(D, M)
b1_init = np.zeros(M)
w2_init = glorot_uniform_initializer(M, K)
b2_init = np.zeros(K)

# define theano variables
thX = T.matrix('X')
thT = T.matrix('T')
w1 = theano.shared(w1_init, 'w1')
b1 = theano.shared(b1_init, 'b1')
w2 = theano.shared(w2_init, 'w2')
b2 = theano.shared(b2_init, 'b2')

# build our model
# ===============

lr = 0.0004
reg = 0.01

# feed forward graph
thZ = T.nnet.relu(thX.dot(w1) + b1)
thY = T.nnet.softmax(thZ.dot(w2) + b2)

cost = -(thT * T.log(thY + 1e-20)).sum() + reg * ((w1 * w1).sum() + (b1 * b1).sum() + (w2 * w2).sum() + (b2 * b2).sum())
prediction = T.argmax(thY, axis=1)

w1_new = w1 - lr * T.grad(cost, w1)
b1_new = b1 - lr * T.grad(cost, b1)
w2_new = w2 - lr * T.grad(cost, w2)
b2_new = b2 - lr * T.grad(cost, b2)

# feed_forward = theano.function(inputs=[thX, thT], output=[thY, cost])
# to do: add momentum, rmsprop or adam

train = theano.function(inputs=[thX, thT],
                        outputs=cost,
                        updates=[(w1, w1_new), (b1, b1_new), (w2, w2_new), (b2, b2_new)])

get_prediction = theano.function(inputs=[thX, thT],
                                 outputs=[cost, prediction])

# train the model for a pre-defined number of epochs
# test set will be used for validation to prevent overfitting
epochs = 10
print('training the model..')
costs = []
for epoch in range(epochs):
    for j in range(num_batches):
        nfirst = j * batch_size
        nlast = nfirst + batch_size
        x_ = Xtrain[nfirst:nlast, :]
        t_ = Ytrain_ind[nfirst:nlast, :]
        train_loss = train(x_, t_)
        costs.append(train_loss)

    val_loss, pred = get_prediction(Xtest, Ytest_ind)
    accuracy = (pred == Ytest).mean() * 100
    print('epoch {0:2d}/{1:d}\t'
          'train_loss={2:7.2f}\t'
          'val_loss={3:7.2f}\t'
          'accuracy={4:6.2f}'.format(epoch + 1, epochs, train_loss, val_loss, accuracy))

plt.plot(costs)
plt.show()
