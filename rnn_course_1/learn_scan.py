import numpy as np
import theano
from theano import tensor as T
from matplotlib import pyplot as plt

# Example 1:
'''
x = T.vector('x')


def square(x_):
    return x_ * x_


y, updates = theano.scan(
    fn=square,
    sequences=x,
    n_steps=x.shape[0]
)

square_op = theano.function(
    inputs=[x],
    outputs=[y]
)

vec = np.array([1, 2, 3, 4, 5])
z = square_op(vec)
print('v', vec.shape, vec)
print('z:', z)
'''

# Example 2:
'''
N = T.iscalar('N')


def recurrence(n, fn_1, fn_2):
    return fn_2, fn_1 + fn_2


outputs, updates = theano.scan(
  fn=recurrence,
  sequences=T.arange(N),
  n_steps=N,
  outputs_info=[0., 1.]
)

fibonacci = theano.function(
  inputs=[N],
  outputs=outputs,
)

o_val = fibonacci(8)

print("output:", o_val[0])
'''

# Example 3:
phi = np.linspace(0, 4*np.pi, 100)
signal = np.sin(phi)
noise = np.random.normal(loc=0, scale=0.2, size=signal.shape)
noisy = signal + noise

x = T.vector('x')
decay = T.scalar('decay')


def smooth(xn, state, coeff):
    return coeff * state + (1-coeff) * xn


y, _ = theano.scan(
    fn=smooth,
    sequences=x,
    n_steps=x.shape[0],
    outputs_info=[np.float64(0)],
    non_sequences=[decay]
)

smooth_op = theano.function(inputs=[x, decay], outputs=y)

clean = smooth_op(noisy, 0.9)

# using tuple unpacking for multiple Axes

lines = []
lines += plt.plot(phi, signal, label='signal')
lines += plt.plot(phi, noisy, label='noisy')
lines += plt.plot(phi, clean, label='clean')
plt.legend(handles=lines)
plt.show()