import theano
from theano import tensor as T

# Example 1:
# ==========
'''
x = T.scalar('x')
y = T.scalar('y')

z = x + y
z = z**2

adder = theano.function(inputs=[x, y], outputs=z)

result = adder(1, 2)

print(z)
print(result)

# theano.printing.pydotprint(z)
'''

# Example 2:
# ==========
'''
c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')

w = A.dot(v)

matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

A_val = np.array([[1, 2], [3, 4]])
v_val = np.array([5, 6])

w_val = matrix_times_vector(A_val, v_val)
print(w_val)
'''

# Example 3:
# ==========
w1 = theano.shared(0, 'w1')
w2 = theano.shared(0, 'w2')
w3 = theano.shared(0, 'w3')

w1_update = w1 + 1
w2_update = w1_update + 1
w3_update = w2_update + 1

update = [(w1, w1_update), (w2, w2_update), (w3, w3_update)]

train = theano.function(inputs=[], outputs=[w1, w2, w3], updates=update)

for i in range(4):
    print(train())
