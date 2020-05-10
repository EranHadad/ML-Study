import numpy as np
import theano
from theano import tensor as T
import pickle
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from util import get_basic_phrases


class SimpleRnn:
    def __init__(self, n_units, word_length, vocabulary_size):
        self.n_units = n_units
        self.word_length = word_length
        self.vocabulary_size = vocabulary_size
        self.activation = None
        self.We = None
        self.Wx = None
        self.h0 = None
        self.Wh = None
        self.bh = None
        self.Wo = None
        self.bo = None
        self.params = None
        # self.thX = None
        # self.thT = None
        # self.Ei = None
        # self.py_x = None
        # self.prediction = None
        self.predict_op = None
        self.forward_op = None

    def set(self, We, Wx, h0, Wh, bh, Wo, bo, activation):
        self.activation = activation
        self.We = theano.shared(We, 'We')
        self.Wx = theano.shared(Wx, 'Wx')
        self.h0 = theano.shared(h0, 'h0')
        self.Wh = theano.shared(Wh, 'Wh')
        self.bh = theano.shared(bh, 'bh')
        self.Wo = theano.shared(Wo, 'Wo')
        self.bo = theano.shared(bo, 'bo')
        self.params = [self.We, self.Wx, self.h0, self.Wh, self.bh, self.Wo, self.bo]

        thX = T.ivector('X')  # T x 1
        thT = T.ivector('T')  # T x 1
        Ei = self.We[thX]  # T x D fmatrix

        def recurrence(x_t, h_t1):
            h_t = self.activation(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            sequences=Ei,
            n_steps=Ei.shape[0],
            outputs_info=[self.h0, None],
        )

        py_x = y[:, 0, :]  # T x V fmatrix
        prediction = T.argmax(py_x, axis=1)  # T x 1 ivector

        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[prediction, py_x],
            allow_input_downcast=True,
        )

        self.forward_op = theano.function(
            inputs=[thX],  # T x 1
            outputs=py_x[-1, :],  # 1 x V
            allow_input_downcast=True,
        )

        return thX, thT, prediction, py_x

    def predict_n(self, X, n_highest):
        py_x = self.forward_op(X)
        ind = np.argpartition(py_x, -n_highest)[-n_highest:]
        ind = ind[np.argsort(py_x[ind])]  # ascending order
        ind = ind[::-1]  # descending order
        prediction_n = ind  # words indices
        confidence_n = py_x[ind]  # words probabilities
        return prediction_n, confidence_n

    def fit(self, X, epochs=10, learning_rate=0.01, activation=T.nnet.relu, show_figure=False):
        V = self.vocabulary_size  # vocabulary size
        D = self.word_length  # word embedding vector length
        M = self.n_units  # hidden layer size

        # initialize model params
        We = SimpleRnn.glorot_uniform_initializer(V, D)
        Wx = SimpleRnn.glorot_uniform_initializer(D, M)
        h0 = np.zeros(M)
        Wh = SimpleRnn.glorot_uniform_initializer(M, M)
        bh = np.zeros(M)
        Wo = SimpleRnn.glorot_uniform_initializer(M, V)
        bo = np.zeros(V)

        # construct forward propagation (prediction)
        thX, thT, prediction, py_x = self.set(We, Wx, h0, Wh, bh, Wo, bo, activation)

        # construct backward propagation (train)
        cost = -T.mean(T.log(py_x[T.arange(thT.shape[0]), thT]))
        updates = SimpleRnn.adam(self.params, cost, learning_rate)

        train_op = theano.function(
            inputs=[thX, thT],
            outputs=[cost, prediction],
            updates=updates,
        )

        N = len(X)
        costs = []
        n_total = sum((len(sentence) + 1) for sentence in X)
        for epoch in range(epochs):
            X = shuffle(X)  # suffle input (list of lists)
            n_correct = 0
            cost = 0
            for j in range(N):
                input_sequence = [0] + X[j]  # 'START' + sentence
                output_sequence = X[j] + [1]  # sentence + 'END'
                c, predictions = train_op(input_sequence, output_sequence)  # prediction is size T x 1
                cost += c
                for pred, out in zip(predictions, output_sequence):
                    if pred == out:
                        n_correct += 1
            accuracy = n_correct / n_total * 100
            print('epoch: {0:3d}/{1:d}, cost: {2:8.3f}, accuracy: {3:6.2f}'.format(epoch + 1, epochs, cost, accuracy))
            costs.append(cost)

        if show_figure:
            plt.plot(costs)
            plt.show()

    @staticmethod
    def glorot_uniform_initializer(n_inputs, n_outputs):
        x = np.sqrt(6 / (n_inputs + n_outputs))
        weights = np.random.uniform(low=-x, high=x, size=(n_inputs, n_outputs))
        return weights

    @staticmethod
    def adam(params, cost, learning_rate):
        # define updates for Adam optimizer:
        beta_1 = 0.9
        beta_2 = 0.99
        epsilon = 1e-07

        grads = T.grad(cost, params)
        moment = [theano.shared(np.zeros(p.get_value().shape)) for p in params]
        mean_square = [theano.shared(np.zeros(p.get_value().shape)) for p in params]

        moment_new = [beta_1 * m + (1 - beta_1) * g for m, g in zip(moment, grads)]
        mean_square_new = [beta_2 * s + (1 - beta_2) * g * g for s, g in zip(mean_square, grads)]

        moment_update = [(m, mnew) for m, mnew in zip(moment, moment_new)]
        mean_square_update = [(s, snew) for s, snew in zip(mean_square, mean_square_new)]
        weight_update = [(p, p - learning_rate * m / (T.sqrt(s) + epsilon)) for p, m, s in zip(params, moment_new,
                                                                                               mean_square_new)]
        updates = moment_update + mean_square_update + weight_update

        return updates

    def save(self, filename='model_info.pckl'):
        # save model parameters to file
        params_dict = {p.__getattribute__('name'): p.get_value() for p in self.params}
        params_dict.update({'activation': self.activation})
        f = open(filename, 'wb')
        pickle.dump(params_dict, f)
        f.close()

    @staticmethod
    def load(filename='model_info.pckl'):
        f = open(filename, 'rb')
        model_info = pickle.load(f)
        f.close()
        We = model_info['We']
        Wx = model_info['Wx']
        h0 = model_info['h0']
        Wh = model_info['Wh']
        bh = model_info['bh']
        Wo = model_info['Wo']
        bo = model_info['bo']
        activation = model_info['activation']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRnn(n_units=M, word_length=D, vocabulary_size=V)
        rnn.set(We, Wx, h0, Wh, bh, Wo, bo, activation)
        return rnn

    def generate(self, word2idx, n_lines=4):
        idx2word = {i: w for w, i in word2idx.items()}
        V = len(word2idx)
        line_count = 0
        while line_count < n_lines:
            # generate first word
            _, py_x = self.predict_op([0])
            pi = py_x[-1, :]
            X = [np.random.choice(V, p=pi)]
            print(idx2word[X[0]], end=" ")
            P = V  # initial value just to enter the loop
            while P != 1:  # P == 1 'END' token
                P, _ = self.predict_op(X)
                P = P[-1]
                if P > 1:  # it's a real word, not 'START'/'END' tokens
                    print(idx2word[P], end=" ")
                    X.append(P)
                elif P == 1:  # 'END' token
                    line_count += 1
                    print('')

    def auto_complete(self, sentence, word2idx):
        n_highest = 3
        idx2word = {i: w for w, i in word2idx.items()}
        n_words = len(sentence)
        for i in range(1, n_words + 1):
            input_sequence = sentence[:i]
            pred, prob = self.predict_n(input_sequence, n_highest=n_highest)
            print('Input:', ' '.join([idx2word.get(key) for key in input_sequence]), '..')
            print('\t', end='')
            for j in range(n_highest):
                print('[{0}-{1:5.1f}%]'.format(idx2word[pred[j]], prob[j] * 100), end='\t')
            print('')
        print('')


def train_phrases():
    sentences, word2idx = get_basic_phrases()
    rnn = SimpleRnn(n_units=30, word_length=30, vocabulary_size=len(word2idx))
    rnn.fit(sentences, epochs=50, learning_rate=0.001, show_figure=True)
    rnn.save()


def generate_phrases():
    sentences, word2idx = get_basic_phrases()
    rnn = SimpleRnn.load()
    V = len(word2idx)
    rnn.generate(word2idx)


def auto_complete(n_sentences):
    sentences, word2idx = get_basic_phrases()
    N = len(sentences)
    rnn = SimpleRnn.load()
    for n in range(n_sentences):
        i = np.random.choice(N)
        rnn.auto_complete(sentences[i], word2idx)


if __name__ == '__main__':
    # train_phrases()
    # generate_phrases()
    auto_complete(n_sentences=4)
