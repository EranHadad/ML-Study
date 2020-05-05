import numpy as np
from matplotlib import pyplot as plt
import theano
from theano import tensor as T

from util import all_parity_pairs_with_sequence_labels
from sklearn.utils import shuffle


def glorot_uniform_initializer(ninputs, noutputs):
    x = np.sqrt(6 / (ninputs + noutputs))
    weights = np.random.uniform(low=-x, high=x, size=(ninputs, noutputs))
    return weights


class SimpleRnn:
    def __init__(self, nUnits):
        self.nUnits = nUnits
        self.f = None
        self.params = None
        self.Wx = None
        self.h0 = None
        self.Wh = None
        self.bh = None
        self.Wo = None
        self.bo = None

    def fit(self, x, t, epochs=100, reg=0.01, lr=0.001, activation=T.tanh, show_figure=False):
        nSamples, seqLen, nInputs = x.shape  # nSamples, seqLen = y.shape
        nOutputs = len(set(t.flatten()))
        nUnits = self.nUnits

        self.f = activation

        thX = T.fmatrix('X')  # seqLen x nInputs
        thT = T.ivector('T')  # targets: 1 x seqLen

        self.Wx = theano.shared(glorot_uniform_initializer(nInputs, nUnits), 'Wx')
        self.h0 = theano.shared(np.zeros(nUnits), 'h0')
        self.Wh = theano.shared(glorot_uniform_initializer(nUnits, nUnits), 'Wh')
        self.bh = theano.shared(np.zeros(nUnits), 'bh')
        self.Wo = theano.shared(glorot_uniform_initializer(nUnits, nOutputs), 'Wo')
        self.bo = theano.shared(np.zeros(nOutputs), 'bo')

        self.params = [self.Wx, self.h0, self.Wh, self.bh, self.Wo, self.bo]

        def recurrence(x_t, h_t1):
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            sequences=thX,
            n_steps=thX.shape[0],
            outputs_info=[self.h0, None],
        )

        py = y[:, 0, :]  # reshape to seqLen x nOutputs
        prediction = T.argmax(py, axis=1)
        cost = -T.mean(T.log(py[T.arange(py.shape[0]), thT]))
        grads = T.grad(cost, self.params)

        # updates for Adam:
        beta_1 = 0.9
        beta_2 = 0.99
        epsilon = 1e-07
        moment = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        mean_square = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        moment_new = [beta_1 * m + (1 - beta_1) * g for m, g in zip(moment, grads)]
        mean_square_new = [beta_2 * s + (1 - beta_2) * g * g for s, g in zip(mean_square, grads)]

        moment_update = [(m, mnew) for m, mnew in zip(moment, moment_new)]
        mean_square_update = [(s, snew) for s, snew in zip(mean_square, mean_square_new)]
        weight_update = [(p, p - lr * m / (T.sqrt(s) + epsilon)) for p, m, s in zip(self.params, moment_new,
                                                                                    mean_square_new)]
        updates = moment_update + mean_square_update + weight_update

        train_op = theano.function(
            inputs=[thX, thT],
            outputs=[cost, prediction],
            updates=updates,
        )

        prediction_op = theano.function(
            inputs=[thX],
            outputs=[prediction],
        )

        costs = []
        for epoch in range(epochs):
            cost = 0
            nCorrect = 0
            for j in range(nSamples):
                c, pred = train_op(x[j], t[j])
                cost += c
                if pred[-1] == t[j, -1]:
                    nCorrect += 1
            costs.append(cost)
            accuracy = nCorrect / nSamples
            print('epoch {0:d}/{1:d}, accuracy={2:4.2f}, cost={3:8.3f}'.format(epoch+1, epochs, accuracy, cost))
            if nCorrect == nSamples:
                break

        if show_figure:
            plt.plot(costs)
            plt.show()


M = 4  # number of recurrent units
B = 4  # number of bits
X, Y = all_parity_pairs_with_sequence_labels(B)
rnn = SimpleRnn(M)
rnn.fit(X, Y, lr=0.01, epochs=200, activation=T.nnet.relu, show_figure=True)
