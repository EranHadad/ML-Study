import numpy as np
import theano
from theano import tensor


def glorot_uniform_initializer(n_inputs, n_outputs):
    x = np.sqrt(6 / (n_inputs + n_outputs))
    weights = np.random.uniform(low=-x, high=x, size=(n_inputs, n_outputs))
    return weights


def adam(params, cost, learning_rate):
    # define updates for Adam optimizer:
    beta_1 = 0.9
    beta_2 = 0.99
    epsilon = 1e-07

    grads = tensor.grad(cost, params)
    moment = [theano.shared(np.zeros(p.get_value().shape)) for p in params]
    mean_square = [theano.shared(np.zeros(p.get_value().shape)) for p in params]

    moment_new = [beta_1 * m + (1 - beta_1) * g for m, g in zip(moment, grads)]
    mean_square_new = [beta_2 * s + (1 - beta_2) * g * g for s, g in zip(mean_square, grads)]

    moment_update = [(m, mnew) for m, mnew in zip(moment, moment_new)]
    mean_square_update = [(s, snew) for s, snew in zip(mean_square, mean_square_new)]
    weight_update = [(p, p - learning_rate * m / (tensor.sqrt(s) + epsilon)) for p, m, s in zip(params, moment_new,
                                                                                                mean_square_new)]
    updates = moment_update + mean_square_update + weight_update

    return updates
