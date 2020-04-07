from mymodule import *
import numpy as np
import matplotlib.pyplot as plt

# script parameters
losscoeffs = [50, 4, -6, 0, 1]
display_loss_function = True
display_graphs = True
run_gradient_descent = False
run_momentum = False
run_Adam = True
# -------------------------------------------

# define loss function
lossfunc = LossFunc(losscoeffs)

# display the loss function
xstart, xstop, npoints = -3, 3, 100
x = np.linspace(xstart, xstop, npoints)
if display_loss_function:
    lossfunc.display(x)

# define optimization algorithm
initial_weight = 3
min_step = 0.001  # early stop
max_iterations = 100

# create gradient descent object
learn_rate = 0.05
grad = GradientDescent(lossfunc=lossfunc, learnrate=learn_rate, weight=initial_weight,
                       minstep=min_step, maxiterations=max_iterations)

# run the optimization algorithm
if run_gradient_descent:
    grad.run()
    grad.printlog()
    if display_graphs:
        grad.plotlog()

# create momentum object
learn_rate = 0.05
momentum_coeff = 0.9
moment = Momentum(lossfunc=lossfunc, learnrate=learn_rate, weight=initial_weight,
                  minstep=min_step, maxiterations=max_iterations, momentcoeff=momentum_coeff)

# run the optimization algorithm
if run_momentum:
    moment.run()
    moment.printlog()
    if display_graphs:
        moment.plotlog()

# create Adam object
learn_rate = 1
beta = 0.9
adam = Adam(lossfunc=lossfunc, learnrate=learn_rate, weight=initial_weight, minstep=min_step,
            maxiterations=max_iterations, momentcoeff=momentum_coeff, beta=beta)

if run_Adam:
    adam.run()
    adam.printlog()
    if display_graphs:
        adam.plotlog()

# show all plots if exist
plt.show(block=True)
