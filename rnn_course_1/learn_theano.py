import theano
from theano import tensor as T

# Example 1:
# ==========

x = T.scalar('x')
y = T.scalar('y')

z = x + y
z = z**2

adder = theano.function(inputs=[x, y], outputs=z)

result = adder(1, 2)

print(z)
print(result)

# theano.printing.pydotprint(z)

# Example 2:
# ==========

c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')

w = A.dot(v)

matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

A_val = np.array([[1, 2], [3, 4]])
v_val = np.array([5, 6])

w_val = matrix_times_vector(A_val, v_val)
print(w_val)

