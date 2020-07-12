import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.datasets.samples_generator import make_blobs
import seaborn as sns
from sklearn.svm import SVC  # "Support vector classifier"
from svm import *
from svm_utils import *

implementation = 'sklearn'  # 'sklearn' # 'study'
C_value = 20  # 1e10

# use seaborn plotting defaults
sns.set()

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=1.0)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='bwr')  # 'brg'
# plt.ylim(-1, 6)
# plt.xlim(-1, 6)
# plt.show(block=False)

# train SVM classifier
if implementation == 'sklearn':
    model = SVC(kernel='rbf', C=C_value)  # 'linear'
    model.fit(X, y)
else:
    y[y == 0] = -1  # SVM expects lables to be equal to -1 for the "negative" class
    model = SVM(X, y, num_of_epochs=10000, lr=1e-3, C=C_value)
    model.fit()

# measure accuracy on training set
p = model.predict(X)
p = p - y.flatten()

# Prediction accuracy should be 1.0 for the training set
print("Accuracy |", len(np.where(p == 0)[0]) / len(p), end="\n\n")

print("number of support vectors:", len(model.support_), sep="\n")

print("support vectors:", model.support_vectors_, sep="\n")

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr')
plot_svc_decision_function(model)
plt.ylim(-1, 6)
plt.xlim(-1, 6)
plt.show()
