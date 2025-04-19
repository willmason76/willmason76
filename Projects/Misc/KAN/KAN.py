import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.linspace(-5, 5, 1000).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(1000, 1)

# Hyperparameters (all fit for a loop with choices)
learning_rate = 0.0001
num_epochs = 1000
num_knots = 10      # Higher knots == more flexibility, but can overfit
k = 3               # Higher k == smoother splines (0 == step, 1 == linear, 2 == quadratic, 3 == cubic [BEST??])

# Data size
input_size = 1
hidden_size = 10
output_size = 1

# Neural Network
class NN:

    # He initialization used to address gradient problems (is this right thing to do???)
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt( 2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt( 2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    # Forward pass
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)

            # Backward pass
            error = y_pred - y
            d_W2 = np.dot(self.a1.T, error)     #gradient of loss with respect to W of output layer
            d_b2 = np.sum(error, axis = 0, keepdims = True)
            d_a1 = np.dot(error, self.W2.T)
            d_z1 = d_a1 * (1 - np.tanh(self.z1) ** 2)
            d_W1 = np.dot(X.T, d_z1)
            d_b1 = np.sum(d_z1, axis = 0, keepdims = True)

            # Clip grads (WHY AM I HAVING THIS PROBLEM??)
            clip_value = 1.0
            d_W2 = np.clip(d_W2, -clip_value, clip_value)
            d_b2 = np.clip(d_b2, -clip_value, clip_value)
            d_W1 = np.clip(d_W1, -clip_value, clip_value)
            d_b1 = np.clip(d_b1, -clip_value, clip_value)

            # Final params
            self.W2 -= learning_rate * d_W2
            self.b2 -= learning_rate * d_b2
            self.W1 -= learning_rate * d_W1
            self.b1 -= learning_rate * d_b1

    def save_model(self, filename):
        np.savez(filename, W1 = self.W1, b1 = self.b1, W2 = self.W2, b2 = self.b2)

# KAN Network
class KAN:
    def __init__(self, input_size, hidden_size, output_size, num_knots = num_knots, k = k):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_knots = num_knots
        self.k = k

        # Initialize knots, coefficients and Splines
        self.knots = np.linspace(-5, 5, num_knots)
        self.coeffs = np.random.randn(hidden_size, num_knots) * 0.01

        # Initialize output layer params
        self.W = np.random.randn(hidden_size, output_size) * np.sqrt( 2. / hidden_size)
        self.b = np.zeros((1, output_size))

    #####   From: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
    def B(self, x, k, i, t):    #(self, x, exponential, iteration of coeffs, knots)

        # First case: If degree of spline is 0, becomes a setp function where result == 1 if x is inside the given knot range, else 0
        if k == 0:
            return 1.0 if t[i] <= x < t[i + 1] else 0.0

        #####   ADDED THIS IN ADDITION TO Scipy DOCS LANGUAGE TO AVOID ERROR   #####
        if (i + k) >= len(t) - 1:
            return 0.0
        #####   ADDED THIS IN ADDITION TO Scipy DOCS LANGUAGE TO AVOID ERROR   #####

        # Second case: to avoid dividing by 0, sets result (c1) == 0 if k == 0
        if t[i + k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i]) / (t[i + k] - t[i]) * self.B(x, k - 1, i, t)

        # Third case: again, to avoid dividing by 0, sets result (c1) == 0 if k == 0
        if t[i + k + 1] == t[i + 1]:
            c2 = 0.0
        else:
            c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * self.B(x, k - 1, i + 1, t)

        return c1 + c2

        # c1 & c2 are the coefficients, and control points on the graph

    def bspline(self, x, t, c, k):
        n = len(t) - k - 1  # n represents number of coeffs
        assert (n >= k + 1) and (len(c) >= n)       # two statements: there are enough knots for given k (exponential degree) & enough control points (c)
        return sum(c[i] * self.B(x, k, i, t) for i in range(n))     # for each iteration of i, calls BSpline function and multiplies by control point c[i], then sums
        # The result can be seen as the BSpline curve at point x, which are linear combinations of basis functions weighted by control points (coefficients)

    # Now does all of the above for hidden layers
    def spline_activation(self, X):
        result = np.zeros((X.shape[0], self.hidden_size))
        for i in range(self.hidden_size):
            result[:, i] = np.array([self.bspline(x, self.knots, self.coeffs[i], self.k) for x in X.flatten()])
        return result

    # Forward pass, which is just the spline calcs plus the regular output layer
    def forward(self, X):
        self.a = self.spline_activation(X)
        return np.dot(self.a, self.W) + self.b

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)

            # Backward pass
            error = y_pred - y
            d_W = np.dot(self.a.T, error)   #gradient of loss with respect to W of output layer
            d_b = np.sum(error, axis = 0, keepdims = True)
            d_a = np.dot(error, self.W.T)

            # Update final spline coefficients
            for i in range(self.hidden_size):
                spline_basis = np.array([self.B(x, self.k, j, self.knots) for x in X.flatten() for j in range(self.num_knots)]).reshape(-1, self.num_knots)
                self.coeffs[i] -= learning_rate * np.dot(d_a[:, i], spline_basis)

            # Clip grads
            clip_value = 1.0
            d_W = np.clip(d_W, -clip_value, clip_value)
            d_b = np.clip(d_b, -clip_value, clip_value)
            self.coeffs = np.clip(self.coeffs, -clip_value, clip_value)

            # For neural network: num_coeffs = hidden_size * num_knots  (would be, and is 100 here)
            # For basic calculations not inside a neural network: num_coeffs = num_knots - k - 1 (would be 6 without the hidden layer = 10)

            # Final params
            self.W -= learning_rate * d_W
            self.b -= learning_rate * d_b

    def save_model(self, filename):
        np.savez(filename, knots = self.knots, coeffs = self.coeffs, W = self.W, b = self.b)

nn = NN(input_size, hidden_size, output_size)
kan = KAN(input_size, hidden_size, output_size)

# Train
nn.train(X, y, learning_rate = learning_rate, epochs = num_epochs)
kan.train(X, y, learning_rate = learning_rate, epochs = num_epochs)

# Save final params for use
nn.save_model('nn_model.npz')
kan.save_model('kan_model.npz')

#####   VISUALIZATION OF RESULTS

# Plot splines
def plot_kan_splines(kan, num_points = 100):
    fig, axs = plt.subplots(2, 5, figsize = (20, 8))
    axs = axs.ravel()

    x = np.linspace(kan.knots.min(), kan.knots.max(), num_points)

    for i in range(kan.hidden_size):
        y = [kan.bspline(xi, kan.knots, kan.coeffs[i], kan.k) for xi in x]
        axs[i].plot(x, y)
        axs[i].set_title(f'Spline Activation: {i + 1}')
        axs[i].axhline(y = 0, color = 'r', linestyle = '--')
        axs[i].axvline(x = 0, color = 'r', linestyle = '--')

    plt.tight_layout()
    plt.show()
plot_kan_splines(kan)

# Plot both model results
# Notes on this: I'm not sure if it is fully informative to only visualize the forward pass between the two
def plot_model_diff(kan):
    plt.figure(figsize = (12, 6))
    plt.scatter(X, y, alpha = 0.3, label = 'Data')
    plt.plot(X, nn.forward(X), label = 'Basic NN', color = 'red')
    plt.plot(X, kan.forward(X), label = f'KAN (knots = {kan.num_knots})', color = 'green')
    plt.legend()
    plt.title('Model Comparison')
    plt.ylim(-5, 5)
    plt.show()
plot_model_diff(kan)

# Printout of final parameters for both models
print(f'\nHere are the Neural Network Final Parameters:\n')
data_nn = np.load('basic_nn_model.npz')
print(data_nn.files)
print(np.round(data_nn['W1'], 4))
print(np.round(data_nn['b1'], 4))
print(np.round(data_nn['W2'], 4))
print(np.round(data_nn['b2'], 4))

print(f'\nHere are the KAN Network Final Parameters:\n')
data = np.load('kan_model.npz')
print(data.files)
print(np.round(data['knots'], 2))
print(np.round(data['coeffs'], 4))
print(np.round(data['W'], 4))
print(np.round(data['b'], 4))
