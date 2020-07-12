import numpy as np
from util import all_parity_pairs

nbit = 4
x, y = all_parity_pairs(nbit)

print(x, y)

odd_ones_ratio = np.sum(y) / y.shape[0]

print(odd_ones_ratio * 100, '%')
