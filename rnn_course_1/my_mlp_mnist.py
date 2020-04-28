import numpy as np
import theano
from theano import tensor as T
from matplotlib import pyplot as plt

from util import get_normalized_data, y2indicator


# To do:
# (1) Early stopping
# (2) Adam

def glorot_uniform_initializer(ninputs, noutputs):
    x = np.sqrt(6 / (ninputs + noutputs))
    weights = np.random.uniform(low=-x, high=x, size=(ninputs, noutputs))
    return weights


class AnnLayer:
    def __init__(self, ninputs, noutputs, an_id, activation=None):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.an_id = an_id
        w_init = glorot_uniform_initializer(ninputs, noutputs)
        b_init = np.zeros(noutputs)
        self.w = theano.shared(w_init, 'w%d' % an_id)
        self.b = theano.shared(b_init, 'b%d' % an_id)
        self.params = [self.w, self.b]
        self.activation = activation

    def forward(self, x):
        linear = x.dot(self.w) + self.b
        if not self.activation:
            y = linear
        elif self.activation == 'relu':
            y = T.nnet.relu(linear)
        elif self.activation == 'softmax':
            y = T.nnet.softmax(linear)
        else:
            print('error: unrecognized activation function')
            exit(1)
        return y


class ANN:
    def __init__(self, hidden_layers_sizes):
        self.hidden_layers_sizes = hidden_layers_sizes
        self.layers = []
        self.params = []

    def fit(self, inputs, targets, validation_data=None, early_stopping=False, lr=4e-4, reg=1e-2, epochs=15, batch_size=500):
        # construct the model
        if inputs.shape[0] != targets.shape[0]:
            print('error: inputs and targets have different number of samples')
            exit(1)
        nsamples, ninputs = inputs.shape
        num_batches = nsamples // batch_size
        noutputs = targets.shape[1]  # require one-hot encoded targets
        num_layers = len(self.hidden_layers_sizes)
        for i in range(num_layers):
            if i == 0:
                n_in = ninputs
            else:
                n_in = self.hidden_layers_sizes[i - 1]
            layer = AnnLayer(n_in, self.hidden_layers_sizes[i], i, 'relu')
            self.layers.append(layer)
        layer = AnnLayer(self.hidden_layers_sizes[i], noutputs, i + 1, 'softmax')  # output layer
        self.layers.append(layer)

        for layer in self.layers:
            self.params.extend(layer.params)

        thX = T.matrix('X')
        thT = T.matrix('T')
        thY = self.forward(thX)

        rcost = reg * T.sum([(p * p).sum() for p in self.params])
        cost = -(thT * T.log(thY + 1e-20)).sum() + rcost
        prediction = T.argmax(thY, axis=1)

        grads = T.grad(cost, self.params)
        updates = [(p, p - lr * g) for p, g in zip(self.params, grads)]

        # define theano functions
        train = theano.function(inputs=[thX, thT], outputs=cost, updates=updates)

        get_prediction = theano.function(inputs=[thX, thT],
                                         outputs=[cost, prediction])

        # train the model
        print('training the model..')

        costs = []
        prev_val_loss = float('inf')
        for epoch in range(epochs):
            for j in range(num_batches):
                nfirst = j * batch_size
                nlast = nfirst + batch_size
                x_ = inputs[nfirst:nlast, :]
                t_ = targets[nfirst:nlast, :]
                train_loss = train(x_, t_)
                costs.append(train_loss)

            if validation_data:
                val_loss, pred = get_prediction(validation_data[0], validation_data[1])
                accuracy = (pred == Ytest).mean() * 100
                print('epoch {0:2d}/{1:d}\t'
                      'train_loss={2:7.2f}\t'
                      'val_loss={3:7.2f}\t'
                      'accuracy={4:6.2f}'.format(epoch + 1, epochs, train_loss, val_loss, accuracy))
                # check for early stopping
                if early_stopping:
                    if val_loss > prev_val_loss:
                        break
                    prev_val_loss = val_loss
            else:
                print('epoch {0:2d}/{1:d}\t'
                      'train_loss={2:7.2f}'.format(epoch + 1, epochs, train_loss))

        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.show()

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y


if __name__ == '__main__':
    # get the MNIST data
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    # apply one-hot encoding to the target vectors
    Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
    Ytest_ind = y2indicator(Ytest).astype(np.float32)

    # instantiate a model and train
    model = ANN([300])
    model.fit(inputs=Xtrain, targets=Ytrain_ind, validation_data=(Xtest, Ytest_ind), early_stopping=True)
