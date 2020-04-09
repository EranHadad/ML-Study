import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class Poly:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.chop_zeros()

    def chop_zeros(self):
        for i in range(len(self.coeffs) - 1, -1, -1):
            if self.coeffs[i] != 0:
                break
        self.coeffs = self.coeffs[:i + 1]

    def degree(self):
        return len(self.coeffs) - 1

    def __str__(self):
        items = []
        for i, x in enumerate(self.coeffs):
            if not x:
                continue
            if i == 0:
                items.append(str(x))
            elif i == 1:
                items.append('{}x'.format(x if x != 1 else ''))
            else:
                items.append('{}x^{}'.format(x if x != 1 else '', i))
        items.reverse()
        res = ' + '.join(items)
        res = res.replace('+ -', '- ')
        return res

    def compute(self, x):
        y = 0
        for i, c in enumerate(self.coeffs):
            y += c * x ** i
        return y

    def derivative(self):
        dp = []
        for i in range(1, len(self.coeffs)):
            dp.append(i * self.coeffs[i])
        return dp


class LossFunc:
    def __init__(self, coeffs):
        self.func = Poly(coeffs)
        self.dfunc = Poly(self.func.derivative())

    def display(self, invec):
        x = invec.reshape(-1, 1)
        y = np.zeros(shape=x.shape, dtype=float)
        dy = np.zeros(shape=x.shape, dtype=float)
        for i in range(len(x)):
            y[i] = self.func.compute(x[i])
            dy[i] = self.dfunc.compute(x[i])
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        fig.suptitle('loss function', fontsize=20)
        ax1.set_facecolor('black')
        ax1.plot(x, y)
        ax1.set_xlabel('x', fontsize=20)
        ax1.set_title('f(x)', fontsize=20)
        ax1.grid()
        ax2.set_facecolor('black')
        ax2.plot(x, dy)
        ax2.set_xlabel('x', fontsize=20)
        ax2.set_title('f\'(x)', fontsize=20)
        ax2.grid()
        plt.show(block=False)


class GradientDescent:
    def __init__(self, lossfunc, learnrate, weight, minstep=0.001, maxiterations=100):
        self.lossfunc = lossfunc
        self.learnrate = learnrate
        self.minstep = minstep
        self.weight = weight
        self.maxiterations = maxiterations
        self.iteration = 0
        self.step = 1000
        columns_names = ['loss', 'step', 'weight']
        self.log = pd.DataFrame(data=np.zeros([maxiterations+1, len(columns_names)]), columns=columns_names)
        self.log.index.name = 'iteration'

    def __next(self):
        self.diffloss = self.lossfunc.dfunc.compute(self.weight)
        self.step = -self.learnrate * self.diffloss
        self.weight += self.step

    def run(self):
        self.__update_log()
        while abs(self.step) > self.minstep and self.iteration < self.maxiterations:
            self.iteration += 1
            self.__next()
            self.__update_log()
        # delete empty lines from log
        self.log.drop(self.log.tail(self.maxiterations-self.iteration).index, inplace=True)

    def __update_log(self):
        self.log.loc[self.iteration, 'loss'] = self.lossfunc.func.compute(self.weight)
        self.log.loc[self.iteration, 'step'] = self.step
        self.log.loc[self.iteration, 'weight'] = self.weight

    def printlog(self):
        decimal_places = 3
        print('\nGradient Descent Performance:')
        print(self.log.round(decimal_places))

    def plotlog(self):
        nepochs = self.log.shape[0]
        ii = np.arange(nepochs)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        fig.suptitle('Gradient Descent Performance', fontsize=20)
        ax1.set_facecolor('black')
        ax1.plot(ii, self.log.loc[:, 'loss'])
        ax1.set_xlabel('Iterations', fontsize=16)
        ax1.set_title('Loss', fontsize=16)
        ax1.grid()
        ax2.set_facecolor('black')
        ax2.plot(ii, self.log.loc[:, 'weight'])
        ax2.set_xlabel('Iterations', fontsize=16)
        ax2.set_title('Weight', fontsize=16)
        ax2.grid()
        plt.show(block=False)


class Momentum:
    def __init__(self, lossfunc, learnrate, weight, minstep=0.001, maxiterations=100, momentcoeff=0.9):
        self.lossfunc = lossfunc
        self.learnrate = learnrate
        self.minstep = minstep
        self.momentcoeff = momentcoeff
        self.last_dl = 0
        self.weight = weight
        self.maxiterations = maxiterations
        self.iteration = 0
        self.step = 1000
        columns_names = ['loss', 'step', 'weight']
        self.log = pd.DataFrame(data=np.zeros([maxiterations+1, len(columns_names)]), columns=columns_names)
        self.log.index.name = 'iteration'

    def __next(self):
        self.diffloss = self.lossfunc.dfunc.compute(self.weight)
        self.last_dl = self.diffloss  # save derivative value for next iteration
        self.step = -self.learnrate * (self.diffloss + self.momentcoeff * self.last_dl)
        self.weight += self.step

    def run(self):
        self.__update_log()
        while abs(self.step) > self.minstep and self.iteration < self.maxiterations:
            self.iteration += 1
            self.__next()
            self.__update_log()
        # delete empty lines from log
        self.log.drop(self.log.tail(self.maxiterations-self.iteration).index, inplace=True)

    def __update_log(self):
        self.log.loc[self.iteration, 'loss'] = self.lossfunc.func.compute(self.weight)
        self.log.loc[self.iteration, 'step'] = self.step
        self.log.loc[self.iteration, 'weight'] = self.weight

    def printlog(self):
        decimal_places = 3
        print('\nMomentum Performance:')
        print(self.log.round(decimal_places))

    def plotlog(self):
        nepochs = self.log.shape[0]
        ii = np.arange(nepochs)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        fig.suptitle('Momentum Performance', fontsize=20)
        ax1.set_facecolor('black')
        ax1.plot(ii, self.log.loc[:, 'loss'])
        ax1.set_xlabel('Iterations', fontsize=16)
        ax1.set_title('Loss', fontsize=16)
        ax1.grid()
        ax2.set_facecolor('black')
        ax2.plot(ii, self.log.loc[:, 'weight'])
        ax2.set_xlabel('Iterations', fontsize=16)
        ax2.set_title('Weight', fontsize=16)
        ax2.grid()
        plt.show(block=False)


class Adam:
    def __init__(self, lossfunc, learnrate, weight, minstep=0.001, maxiterations=100, momentcoeff=0.9, beta=0.9):
        self.lossfunc = lossfunc
        self.learnrate = learnrate
        self.minstep = minstep
        self.momentcoeff = momentcoeff
        self.beta = beta
        self.weight = weight
        self.maxiterations = maxiterations
        self.moment_state = 0
        self.G_state = 0
        self.iteration = 0
        self.diffloss = 0
        self.step = 1000
        columns_names = ['loss', 'step', 'weight', 'diffloss', 'moment', 'G_term']
        self.log = pd.DataFrame(data=np.zeros([maxiterations+1, len(columns_names)]), columns=columns_names)
        self.log.index.name = 'iteration'

    def __next(self):
        self.diffloss = self.lossfunc.dfunc.compute(self.weight)
        self.moment_state = self.momentcoeff * self.moment_state + (1-self.momentcoeff) * self.diffloss
        self.G_state = self.beta * self.G_state + (1 - self.beta) * self.diffloss**2
        self.step = -self.learnrate * self.moment_state / (math.sqrt(self.G_state)+1e-10)
        self.weight += self.step

    def run(self):
        self.__update_log()
        while abs(self.step) > self.minstep and self.iteration < self.maxiterations:
            self.iteration += 1
            self.__next()
            self.__update_log()
        # delete empty lines from log
        self.log.drop(self.log.tail(self.maxiterations-self.iteration).index, inplace=True)

    def __update_log(self):
        self.log.loc[self.iteration, 'loss'] = self.lossfunc.func.compute(self.weight)
        self.log.loc[self.iteration, 'step'] = self.step
        self.log.loc[self.iteration, 'weight'] = self.weight
        self.log.loc[self.iteration, 'diffloss'] = self.diffloss
        self.log.loc[self.iteration, 'moment'] = self.moment_state
        self.log.loc[self.iteration, 'G_term'] = self.G_state

    def printlog(self):
        decimal_places = 3
        print('\nAdam Performance:')
        print(self.log.round(decimal_places))

    def plotlog(self):
        nepochs = self.log.shape[0]
        ii = np.arange(nepochs)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        fig.suptitle('Adam Performance', fontsize=20)
        ax1.set_facecolor('black')
        ax1.plot(ii[1:], self.log.loc[1:, 'diffloss'], label='derivative')
        ax1.plot(ii[1:], self.log.loc[1:, 'moment'], label='moment')
        vsqrt = np.vectorize(math.sqrt)
        ax1.plot(ii[1:], vsqrt(self.log.loc[1:, 'G_term']), label='rms')
        ax1.set_xlabel('Iterations', fontsize=16)
        ax1.legend(fontsize=16)
        ax1.grid()

        ax2.set_facecolor('black')
        ax2.plot(ii[1:], self.log.loc[1:, 'step'])
        ax2.set_xlabel('Iterations', fontsize=16)
        ax2.set_title('Step', fontsize=16)
        ax2.grid()

        plt.show(block=False)
