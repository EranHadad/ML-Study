import numpy as np
import pandas as pd

maxiterations = 4
columns = ['loss', 'step', 'weight']
ncolumns = len(columns)
nrows = maxiterations + 1

df = pd.DataFrame(data=np.zeros([nrows, ncolumns]), columns=columns)
df.index.name = 'iteration'

df.loc[0, 'loss'] = 7
print(df)

df.drop(df.tail(2).index, inplace=True)
print(df)

'''
loss, step = self.__next()
print("Iteration: {itr:3d}/{max_itr}, Loss: {loss:8.3f}, Step: {step:8.3f}, "
      "weight: {weight:8.3f}".format(itr=itr, max_itr=self.maxiterations,
                                     loss=loss, step=step, weight=self.weight))
'''
