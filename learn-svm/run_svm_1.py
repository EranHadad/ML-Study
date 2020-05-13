from svm import *

data = np.genfromtxt('files/data.csv', dtype=float, delimiter=',')
np.random.shuffle(data)

train_y = data[:, 0]
train_x = data[:, 1:]

clf = SVM(train_x, train_y, num_of_epochs=10000, lr=1e-3, C=30)
clf.fit()

p = clf.predict(train_x)
p = p - train_y.flatten()

# Prediction accuracy should be 1.0 for the training set
print("Accuracy |", len(np.where(p == 0)[0]) / len(p))
