import tensorflow as tf

'''
BATCH_SIZE = 2

A = tf.data.Dataset.range(1, 7)
print(list(A.as_numpy_iterator()))
A = A.map(lambda x: tf.fill([x], x))

print(A.element_spec)
print(list(A.as_numpy_iterator()))

maxlen = 0
for t in A:
    currlen = t.shape[0]
    if currlen > maxlen:
        maxlen = currlen

B = A.padded_batch(BATCH_SIZE, padded_shapes=maxlen, drop_remainder=True)

print(B.element_spec)
print(list(B.as_numpy_iterator()))

for ex in B:
    print(ex.numpy())
'''

BATCH_SIZE = 2

A = tf.data.Dataset.range(1, 8)
# print(list(A.as_numpy_iterator()))
A = A.map(lambda x: tf.fill([x], x))

L = tf.data.Dataset.from_tensor_slices([0, 1, 2, 1, 0, 2, 0])
A = tf.data.Dataset.zip((A, L))

# print(A.element_spec)
# print(list(A.as_numpy_iterator()))


maxlen = 0
for t, _ in A:
    currlen = t.shape[0]
    if currlen > maxlen:
        maxlen = currlen

B = A.padded_batch(BATCH_SIZE, drop_remainder=True, padded_shapes=([None], []))

print(B.element_spec)
for ex in B:
    print(ex)

