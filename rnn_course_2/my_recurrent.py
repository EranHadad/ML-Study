import numpy as np
import theano
from theano import tensor
from my_utils import glorot_uniform_initializer


class RU:
    def __init__(self, n_inputs, n_units, activation):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.activation = activation

        Wxh = glorot_uniform_initializer(n_inputs, n_units)
        Whh = glorot_uniform_initializer(n_units, n_units)
        bh  = np.zeros(n_units)
        h0  = np.zeros(n_units)

        self.Wxh = theano.shared(Wxh, 'Wxh')
        self.Whh = theano.shared(Whh, 'Whh')
        self.bh  = theano.shared(bh, 'bh')
        self.h0  = theano.shared(h0, 'h0')
        self.params = [self.Wxh, self.Whh, self.bh, self.h0]

    def recurrence(self, x_t, h_t1):
        h_t = self.activation(x_t.dot(self.Wxh) + h_t1.dot(self.Whh) + self.bh)
        return h_t

    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            n_steps=x.shape[0],
            outputs_info=[self.h0],
        )
        return h


class RRU:
    def __init__(self, n_inputs, n_units, activation):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.activation = activation

        Wxh = glorot_uniform_initializer(n_inputs, n_units)
        Whh = glorot_uniform_initializer(n_units, n_units)
        bh  = np.zeros(n_units)
        h0  = np.zeros(n_units)
        Wxz = glorot_uniform_initializer(n_inputs, n_units)
        Whz = glorot_uniform_initializer(n_units, n_units)
        bz  = np.zeros(n_units)

        self.Wxh = theano.shared(Wxh, 'Wxh')
        self.Whh = theano.shared(Whh, 'Whh')
        self.bh  = theano.shared(bh, 'bh')
        self.h0  = theano.shared(h0, 'h0')
        self.Wxz = theano.shared(Wxz, 'Wxz')
        self.Whz = theano.shared(Whz, 'Whz')
        self.bz  = theano.shared(bz, 'bz')
        self.params = [self.Wxh, self.Whh, self.bh, self.h0, self.Wxz, self.Whz, self.bz]

    def recurrence(self, x_t, h_t1):
        hhat_t = self.activation(x_t.dot(self.Wxh) + h_t1.dot(self.Whh) + self.bh)
        z_t = tensor.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
        h_t = (1 - z_t) * h_t1 + z_t * hhat_t
        return h_t

    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            n_steps=x.shape[0],
            outputs_info=[self.h0],
        )
        return h


class GRU:
    def __init__(self, n_inputs, n_units, activation):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.activation = activation

        Wxr = glorot_uniform_initializer(n_inputs, n_units)
        Whr = glorot_uniform_initializer(n_units, n_units)
        br = np.zeros(n_units)
        Wxz = glorot_uniform_initializer(n_inputs, n_units)
        Whz = glorot_uniform_initializer(n_units, n_units)
        bz = np.zeros(n_units)
        Wxh = glorot_uniform_initializer(n_inputs, n_units)
        Whh = glorot_uniform_initializer(n_units, n_units)
        bh  = np.zeros(n_units)
        h0  = np.zeros(n_units)

        self.Wxr = theano.shared(Wxr, 'Wxr')
        self.Whr = theano.shared(Whr, 'Whr')
        self.br = theano.shared(br, 'br')
        self.Wxz = theano.shared(Wxz, 'Wxz')
        self.Whz = theano.shared(Whz, 'Whz')
        self.bz = theano.shared(bz, 'bz')
        self.Wxh = theano.shared(Wxh, 'Wxh')
        self.Whh = theano.shared(Whh, 'Whh')
        self.bh  = theano.shared(bh, 'bh')
        self.h0  = theano.shared(h0, 'h0')

        self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh, self.h0]

    def recurrence(self, x_t, h_t1):
        r_t = tensor.nnet.sigmoid(x_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
        hhat_t = self.activation(x_t.dot(self.Wxh) + (r_t * h_t1).dot(self.Whh) + self.bh)
        z_t = tensor.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
        h_t = (1 - z_t) * h_t1 + z_t * hhat_t
        return h_t

    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            n_steps=x.shape[0],
            outputs_info=[self.h0],
        )
        return h


class LSTM:
    def __init__(self, n_inputs, n_units, activation):
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.activation = activation

        Wxi = glorot_uniform_initializer(n_inputs, n_units)
        Whi = glorot_uniform_initializer(n_units, n_units)
        Wci = glorot_uniform_initializer(n_units, n_units)
        bi = np.zeros(n_units)

        Wxf = glorot_uniform_initializer(n_inputs, n_units)
        Whf = glorot_uniform_initializer(n_units, n_units)
        Wcf = glorot_uniform_initializer(n_units, n_units)
        bf = np.zeros(n_units)

        Wxc = glorot_uniform_initializer(n_inputs, n_units)
        Whc = glorot_uniform_initializer(n_units, n_units)
        bc = np.zeros(n_units)

        Wxo = glorot_uniform_initializer(n_inputs, n_units)
        Who = glorot_uniform_initializer(n_units, n_units)
        Wco = glorot_uniform_initializer(n_units, n_units)
        bo = np.zeros(n_units)

        h0  = np.zeros(n_units)
        c0 = np.zeros(n_units)

        self.Wxi = theano.shared(Wxi, 'Wxi')
        self.Whi = theano.shared(Whi, 'Whi')
        self.Wci = theano.shared(Wci, 'Wci')
        self.bi = theano.shared(bi, 'bi')

        self.Wxf = theano.shared(Wxf, 'Wxf')
        self.Whf = theano.shared(Whf, 'Whf')
        self.Wcf = theano.shared(Wcf, 'Wcf')
        self.bf = theano.shared(bf, 'bf')

        self.Wxc = theano.shared(Wxc, 'Wxc')
        self.Whc = theano.shared(Whc, 'Whc')
        self.bc = theano.shared(bc, 'bc')

        self.Wxo = theano.shared(Wxo, 'Wxo')
        self.Who = theano.shared(Who, 'Who')
        self.Wco = theano.shared(Wco, 'Wco')
        self.bo = theano.shared(bo, 'bo')

        self.h0 = theano.shared(h0, 'h0')
        self.c0 = theano.shared(c0, 'c0')

        self.params = [self.Wxi, self.Whi, self.Wci, self.bi,
                       self.Wxf, self.Whf, self.Wcf, self.bf,
                       self.Wxc, self.Whc, self.bc,
                       self.Wxo, self.Who, self.Wco, self.bo,
                       self.h0, self.c0]

    def recurrence(self, x_t, h_t1, c_t1):
        i_t = tensor.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        f_t = tensor.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        chat_t = tensor.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        c_t = f_t * c_t1 + i_t * chat_t
        o_t = tensor.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o_t * tensor.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            n_steps=x.shape[0],
            outputs_info=[self.h0, self.c0],
        )
        return h
