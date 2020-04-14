import numpy as np


# implemention for activation function of f(x)=x
class Activation:
    @staticmethod
    def derivative(y):
        return 1

    @staticmethod
    def activate(x):
        return x


# implemention for logistic activation function y = f(x) = 1/(1+exp(-x))
# f'(x) = f(x) * (1-f(x)) = y * (1-y)
class Logistic(Activation):
    @staticmethod
    def derivative(y):
        return np.multiply(y, 1-y)
        # y = f(x) - activation output computed with x as an input
        # derivative(x) = activate(x) * (1-activate(x))

    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, noutputs=None, activation=None, init_range=0.1):
        self.noutputs = noutputs
        if activation == 'logistic':
            self.activation = Logistic()
        else:
            self.activation = Activation()
        self.ninputs = None
        self.init_range = init_range
        self.weights = None  # dimensions=(ninputs, noutputs) will determine on model fit
        self.biases = np.random.uniform(low=-init_range, high=init_range, size=(1, noutputs))
        self.outputs = None
        self.errors = None
        # print('Class Layer was instantiated with {nout} outputs'.format(nout=noutputs))

    def process_outputs(self, x):
        if x.shape[1] != self.ninputs:
            print('error in process_outputs(): can not perform dot product due to shape mismathch')
        self.outputs = np.dot(x, self.weights) + self.biases
        self.outputs = self.activation.activate(self.outputs)


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
        self.learning_rate = None
        self.loss = None

    def fit(self, x=None, y=None, epochs=1, batch_size=None, learning_rate=0.01):

        # initialize net
        self.__init_model(x, y, batch_size, learning_rate)

        # perform the fit
        for epoch in range(epochs):
            
            self.loss = 0

            for batch_index in range(self.nbatches):
                nfrom = batch_index * self.batch_size
                nlast = (batch_index+1) * self.batch_size
                inputs = x[nfrom:nlast, :]
                targets = y[nfrom:nlast, :]
                self.__forward_propagate(inputs)  # compute outputs foreach layer
                self.__back_propagate(targets)  # compute errors for each layer
                self.__update_coefficients(inputs)

            # print current epoch data (epoch number, loss function..)
            print('Epoch {ind:d}/{total:d} - loss: {loss:.4f}'.format(ind=epoch+1, total=epochs, loss=self.loss))

    def predict(self, x=None):
        self.__forward_propagate(x)
        return self.layers[-1].outputs

    def __init_model(self, x, y, batch_size, learning_rate):

        self.learning_rate = learning_rate

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
                layer.ninputs = self.layers[i-1].noutputs

            print('layer {index}: {nin} inputs -> {nout} outputs'.format(index=i,
                                                                         nin=layer.ninputs,
                                                                         nout=layer.noutputs))
            layer.weights = np.random.uniform(low=-layer.init_range,
                                              high=layer.init_range,
                                              size=(layer.ninputs, layer.noutputs))

    def __forward_propagate(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.process_outputs(x)
            else:
                layer.process_outputs(self.layers[i-1].outputs)

    def __back_propagate(self, targets):
        for i in reversed(range(self.nlayers)):
            y = self.layers[i].outputs
            if i == (self.nlayers-1):
                self.layers[i].errors = y - targets
                self.loss += np.sum(self.layers[i].errors ** 2) / 2 / self.batch_size  # also update the loss value
            else:
                self.layers[i].errors = np.dot(self.layers[i+1].errors, self.layers[i+1].weights.transpose())
            self.layers[i].errors = np.multiply(self.layers[i].errors, self.layers[i].activation.derivative(y))

    def __update_coefficients(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                inputs = x
            else:
                inputs = self.layers[i-1].outputs
            dloss_dw = (1/self.batch_size) * np.dot(inputs.transpose(), layer.errors)
            layer.weights -= self.learning_rate * dloss_dw
            dloss_dbias = np.mean(layer.errors, axis=0)
            layer.biases -= self.learning_rate * dloss_dbias
