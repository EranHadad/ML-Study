import numpy as np
import theano
from theano import tensor as T
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from util import get_basic_phrases
from my_recurrent import RU, RRU, GRU, LSTM
from my_utils import glorot_uniform_initializer, adam


class RNN:
    def __init__(self, word_length, hidden_layer_sizes, vocabulary_size):
        self.word_length = word_length
        self.hidden_layer_sizes = hidden_layer_sizes
        self.vocabulary_size = vocabulary_size
        self.We = None  # word embeddings
        self.hidden_layers = None
        self.Wo = None  # logistic regression
        self.bo = None  # logistic regression
        self.params = None
        self.predict_op = None
        self.forward_op = None

    def predict_n(self, X, n_highest):
        py_x = self.forward_op(X)
        ind = np.argpartition(py_x, -n_highest)[-n_highest:]
        ind = ind[np.argsort(py_x[ind])]  # ascending order
        ind = ind[::-1]  # descending order
        prediction_n = ind  # words indices
        confidence_n = py_x[ind]  # words probabilities
        return prediction_n, confidence_n

    def fit(self, X, epochs=10, learning_rate=0.01, activation=T.nnet.relu, RecurrentUnit=GRU, show_figure=False):
        V = self.vocabulary_size  # vocabulary size
        D = self.word_length  # word embedding vector length

        # initialize model params
        We = glorot_uniform_initializer(V, D)
        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo
        Wo = glorot_uniform_initializer(Mi, V)
        bo = np.zeros(V)

        # construct forward propagation (prediction)
        self.We = theano.shared(We, 'We')
        self.Wo = theano.shared(Wo, 'Wo')
        self.bo = theano.shared(bo, 'bo')

        self.params = [self.We, self.Wo, self.bo]
        for layer in self.hidden_layers:
            self.params += layer.params

        thX = T.ivector('X')  # T x 1
        thT = T.ivector('T')  # T x 1
        Z = self.We[thX]  # T x D fmatrix

        for ru in self.hidden_layers:
            Z = ru.output(Z)

        py_x = T.nnet.softmax(Z.dot(self.Wo + self.bo))  # T x V fmatrix
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

        # construct backward propagation (train)
        cost = -T.mean(T.log(py_x[T.arange(thT.shape[0]), thT]))
        updates = adam(self.params, cost, learning_rate)

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

    def generate(self, word2idx, n_lines=4):
        idx2word = {i: w for w, i in word2idx.items()}
        V = len(word2idx)
        _, py_x = self.predict_op([0])
        pi = py_x[-1, :]
        line_count = 0
        while line_count < n_lines:
            # generate first word
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


if __name__ == '__main__':
    # train
    sentences, word2idx = get_basic_phrases()
    rnn = RNN(word_length=30, hidden_layer_sizes=[30], vocabulary_size=len(word2idx))
    rnn.fit(sentences, epochs=50, learning_rate=0.001, show_figure=True)

    # auto-complete
    n_sentences = 4
    N = len(sentences)
    for n in range(n_sentences):
        i = np.random.choice(N)
        rnn.auto_complete(sentences[i], word2idx)

    # generate sentences
    rnn.generate(word2idx)
