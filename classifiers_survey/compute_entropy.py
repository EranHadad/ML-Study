from matplotlib import pyplot as plt
from math import log2

# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]

# plot of distributions
# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
print('P=%.3f Q=%.3f' % (sum(p), sum(q)))

# plot first distribution
plt.subplot(2, 1, 1)
plt.bar(events, p)

# plot second distribution
plt.subplot(2, 1, 2)
plt.bar(events, q)

# show the plot
plt.show()


# calculate cross-entropy
def cross_entropy(p_, q_):
    return (-1) * sum([p_[i] * log2(q_[i]) for i in range(len(p_))])


# calculate cross-entropy
def entropy(p_):
    return (-1) * sum([p_[i] * log2(p_[i]) for i in range(len(p_))])


# calculate KL divergence
def kl_divergence(p_, q_):
    return (-1) * sum([p_[i] * log2(q_[i]/p_[i]) for i in range(len(p_))])


# calculate H(P)
print('H(P): %.3f bits' % entropy(p))
# calculate kl divergence KL(P || Q)
print('KL(P || Q): %.3f bits' % kl_divergence(p, q))
# calculate cross entropy H(P, Q)
print('H(P, Q): %.3f bits' % cross_entropy(p, q))
# calculate cross entropy using H(P, Q) = H(P) + KL(P || Q)
print('H(P, Q) = H(P) + KL(P || Q): %.3f bits' % (entropy(p) + kl_divergence(p, q)))

