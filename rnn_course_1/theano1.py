import numpy as np
import theano
from theano import tensor as T

x = theano.shared(20.0, 'x')

cost = x * x + x + 1
learn_rate = 0.3
step = -learn_rate * T.grad(cost, x)
x_new = x + step

train = theano.function(inputs=[], outputs=[cost, step], updates=[(x, x_new)])

for i in range(50):
    cost_val, step_val = train()
    print('itr={0:2d}\tcost={1:8.4f}\tstep={2:8.4f}'.format(i, cost_val, step_val))
    if abs(step_val) < 1e-3:
        print('converged after {itr:d} iterations'.format(itr=i+1))
        break

xopt = x.get_value()
print('x optimal value: {val:.3f}'.format(val=xopt))
