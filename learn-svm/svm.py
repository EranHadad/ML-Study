import numpy as np


class SVM:
    def __init__(self, X, y, kernel='linear', num_of_epochs=10000, lr=1e-3, C=100):

        self.X = X
        self.y = y
        self.num_of_epochs = num_of_epochs
        self.lr = lr
        self.C = C
        self.support_ = None
        self.support_vectors_ = None
        self.kernel = kernel

        self.features = self.compute_features(X)

        # Initialize normal vector
        self.w = np.ones(len(self.features[0]))

    def compute_features(self, X):
        num_of_samples = X.shape[0]

        if self.kernel == 'rbf':
            num_of_landmarks = self.X.shape[0]
            features = np.zeros(shape=(num_of_samples, num_of_landmarks))
            for i in range(num_of_samples):
                curr_sample = X[i, :]
                for j in range(num_of_landmarks):  # landmark
                    features[i, j] = SVM.compute_rbf(curr_sample, self.X[j, :])
        else:  # (default) linear
            features = X

        # Add column vector of ones for computational convenience
        features = np.column_stack((np.ones(num_of_samples), features))

        return features

    @staticmethod
    def compute_rbf(x1, x2):
        sigma_sqr = 1  # TO DO: edit this part of code
        x = x1 - x2
        return np.exp(-0.5 * x.dot(x) / sigma_sqr)

    def distances(self, with_lagrange=True):
        distances = self.y * (np.dot(self.features, self.w)) - 1

        # get distance from the current decision boundary
        # by considering 1 width of margin

        if with_lagrange:  # if lagrange multiplier considered
            # if distance is more than 0
            # sample is not on the support vector
            # Lagrange multiplier will be 0
            distances[distances > 0] = 0

        return distances

    def update_support_vectors(self, distances):
        self.support_ = np.where(distances < 0)[0]
        self.support_vectors_ = self.X[self.support_, :]

    def get_cost_grads(self):

        distances = self.distances()

        # Get current cost
        L = 0.5 * np.dot(self.w, self.w) - self.C * np.sum(distances)

        self.update_support_vectors(distances)

        # constraint term
        di = np.zeros(len(self.w))
        for ind in self.support_:
            di += self.y[ind] * self.features[ind]

        # weight minimization term
        dw = np.zeros(len(self.w))
        dw[1:] = self.w[1:]  # skip first element (keep it zero, updating bias term differently)

        dw -= self.C * di  # C = 1 / lambda, where lambda is the regularization parameter.
        # large C gives more weight to the constraint and possibly will result with a narrower margin but with less
        # misscalssified samples or samples that reside indide the margin (soft margin)

        # return L, dw / len(self.X)  # original
        return L, dw

    def fit(self):
        for i in range(self.num_of_epochs):
            L, dw = self.get_cost_grads()
            self.w = self.w - self.lr * dw
            if i % 1000 == 0:
                print(i, ' | ', L)

    def decision_function(self, X):
        feature = self.compute_features(X)
        return np.matmul(feature, self.w)

    def predict(self, X):
        margin = self.decision_function(X)
        return np.sign(margin)
