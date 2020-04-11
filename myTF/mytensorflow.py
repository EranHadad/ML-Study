import numpy as np


class Layer:
    def __init__(self, noutputs=None, init_range=0.1):
        self.noutputs = noutputs
        self.ninputs = None
        self.init_range = init_range
        self.weights = None  # dimensions=(ninputs, noutputs) will determine on model fit
        self.biases = np.random.uniform(low=-init_range, high=init_range, size=(1, noutputs))
        # print('Class Layer was instantiated with {nout} outputs'.format(nout=noutputs))


# sequential model with L2-norm loss function and gradient-descent optimizer
class Model:
    def __init__(self, layers_list=None):
        self.layers = layers_list
        self.nlayers = len(layers_list)
        self.noutputs = layers_list[self.nlayers - 1].noutputs
        self.ninputs = None
        self.nsamples = None
        self.batch_size = None
        self.nbatches = None

    def fit(self, x=None, y=None, epochs=1, batch_size=None):

        # initialize net
        self.__init_model(x, y, batch_size)

        # perform the fit
        for epoch in range(epochs):
            for batch_index in range(self.nbatches):
                nfrom = batch_index * self.batch_size
                nlast = (batch_index+1) * self.batch_size
                inputs = x[nfrom:nlast, :]
                targets = y[nfrom:nlast, :]
                self.__forward_propagate(inputs, targets)  # also compute errors

        # for each epoch
        #   for each batch
        #       self.__forward_propagate()
        #       self.__back_propagate()
        #   print results (optionaly: save results to log)

    def __init_model(self, x, y, batch_size):
        self.ninputs = x.shape[1]

        if self.noutputs != y.shape[1]:
            print('targets vector y has incompatible number of outputs, expected {nout:d}'.format(nout=self.noutputs))
            exit(1)

        if x.shape[0] == y.shape[0]:
            self.nsamples = x.shape[0]
        else:
            print('targets vector y and inputs vector have different number of samples')
            exit(1)

        if not batch_size or batch_size > self.nsamples:
            self.batch_size = self.nsamples
        else:
            self.batch_size = batch_size
        self.nbatches = int(self.nsamples / self.batch_size)
        print('batch size = {batchsize:d}'.format(batchsize=self.batch_size))

        for i, layer in enumerate(self.layers):

            if i == 0:
                layer.ninputs = self.ninputs
            else:
                layer.ninputs = self.layers[i - 1].noutputs

            print('layer {index}: {nin} inputs -> {nout} outputs'.format(index=i,
                                                                         nin=layer.ninputs,
                                                                         nout=layer.noutputs))
            layer.weights = np.random.uniform(low=-layer.init_range,
                                              high=layer.init_range,
                                              size=(layer.ninputs, layer.noutputs))

