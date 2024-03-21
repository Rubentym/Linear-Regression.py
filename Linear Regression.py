import numpy as np
import matplotlib.pyplot as plt

# Generate random data points
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to the feature matrix
X_b = np.c_[np.ones((100, 1)), X]

# Implement Gradient Descent for linear regression
eta = 0.1  # learning rate
n_iterations = 1000
theta = np.random.randn(2, 1)
for iteration in range(n_iterations):
    gradients = 2 / 100 * X_b.T.dot(X_b.dot(theta) - y)
    theta -= eta * gradients

# Plot the original data and the linear regression line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Descent for Linear Regression')
plt.show()
