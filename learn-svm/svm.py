import numpy as np


class SVM:
    def __init__(self, X, y, num_of_epochs, lr, C):

        self.X = X
        self.y = y
        self.num_of_epochs = num_of_epochs
        self.lr = lr
        self.C = C
        self.support_vectors_ = None

        # Add column vector of ones for computational convenience
        self.X = np.column_stack((np.ones(len(X)), X))

        # Initialize normal vector
        self.w = np.ones(len(self.X[0]))

    def distances(self, w, with_lagrange=True):
        distances = self.y * (np.dot(self.X, w)) - 1

        # get distance from the current decision boundary
        # by considering 1 width of margin

        if with_lagrange:  # if lagrange multiplier considered
            # if distance is more than 0
            # sample is not on the support vector
            # Lagrange multiplier will be 0
            distances[distances > 0] = 0

        return distances

    def update_support_vectors(self, distances):
        indices = distances < 0
        self.support_vectors_ = self.X[indices, 1:]

    def get_cost_grads(self, X, w, y):

        distances = self.distances(w)

        self.update_support_vectors(distances)

        # Get current cost
        L = 1 / 2 * np.dot(w, w) - self.C * np.sum(distances)

        dw = np.zeros(len(w))

        for ind, d in enumerate(distances):
            if d == 0:  # if sample is not on the support vector
                di = w  # (alpha * y[ind] * X[ind]) = 0
            else:
                # (alpha * y[ind] * X[ind]) = y[ind] * X[ind]
                di = w - (self.C * y[ind] * X[ind])
            dw += di
        return L, dw / len(X)

    def fit(self):
        for i in range(self.num_of_epochs):
            L, dw = self.get_cost_grads(self.X, self.w, self.y)
            self.w = self.w - self.lr * dw
            if i % 1000 == 0:
                print(i, ' | ', L)

    def decision_function(self, X):
        X = np.column_stack((np.ones(len(X)), X))
        # return X @ self.w
        return np.matmul(X, self.w)

    def predict(self, X):
        margin = self.decision_function(X)
        return np.sign(margin)



