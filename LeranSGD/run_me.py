from mymodule import *
import numpy as np
import matplotlib.pyplot as plt

# script parameters
losscoeffs = [50, 4, -6, 0, 1]
display_loss_function = False
run_gradient_descent = True
run_momentum = True
# -------------------------------------------

# define loss function
lossfunc = LossFunc(losscoeffs)

# display the loss function
xstart, xstop, npoints = -3, 3, 100
x = np.linspace(xstart, xstop, npoints)
if display_loss_function:
    lossfunc.display(x)

# define optimization algorithm
learn_rate = 0.05
initial_weight = 0.7
min_step = 0.001  # early stop
max_iterations = 100

# create gradient descent object
grad = GradientDescent(lossfunc=lossfunc, learnrate=learn_rate, weight=initial_weight,
                       minstep=min_step, maxiterations=max_iterations)

# run the optimization algorithm
if run_gradient_descent:
    grad.run()
    grad.printlog()
    grad.plotlog()

# create momentum object
momentum_coeff = 0.9
moment = Momentum(lossfunc=lossfunc, learnrate=learn_rate, weight=initial_weight,
                  minstep=min_step, maxiterations=max_iterations, momentcoeff=momentum_coeff)

# run the optimization algorithm
if run_momentum:
    moment.run()
    moment.printlog()
    moment.plotlog()

# show all plots if exist
plt.show(block=True)
