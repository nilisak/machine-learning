import numpy as np


class NPLinear:
    def __init__(self, in_channels, out_channels):
        # Initialize weights and biases
        limit = np.sqrt(1 / in_channels)
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels))
        self.b = np.zeros(out_channels)

        # Initialize gradients
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)

    def forward(self, x):
        # Compute the forward pass
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad_output):
        # Compute gradients
        self.W_grad = grad_output.T @ self.x
        self.b_grad = np.sum(grad_output, axis=0)
        return self.x

    def gd_update(self, lr):
        # Update weights and biases using gradient descent
        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad

    def zero_grad(self):
        # Reset gradients to zero
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)

    # Additional method to store the input (for use in backward pass)
    def save_input(self, x):
        self.x = x


# Example usage
np_linear = NPLinear(10, 5)
x = np.random.randn(3, 10)  # batch of 3 samples, 10 input channels
output = np_linear.forward(x)
np_linear.save_input(x)

# Assuming some loss and its gradient w.r.t. output
grad_output = np.random.randn(3, 5)  # same shape as output
np_linear.backward(grad_output)
np_linear.gd_update(lr=0.01)
