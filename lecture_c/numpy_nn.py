import numpy as np
import matplotlib.pyplot as plt


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
        return grad_output @ self.W

    def gd_update(self, lr):
        # Update weights and biases using gradient descent
        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad

    def zero_grad(self):
        # Reset gradients to zero
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)


class NPMSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        size = np.size(self.pred)
        return 2 * (self.pred - self.target) / size


class NPReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input < 0] = 0
        return grad_input


class NPModel:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NPLinear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No ReLU after last layer
                self.layers.append(NPReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def gd_update(self, lr):
        for layer in self.layers:
            if isinstance(layer, NPLinear):
                layer.gd_update(lr)

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, NPLinear):
                layer.zero_grad()


N_TRAIN = 100
N_TEST = 1000
SIGMA_NOISE = 0.1

np.random.seed(0xDEADBEEF)
x_train = np.random.uniform(low=-np.pi, high=np.pi, size=N_TRAIN)[:, None]
y_train = np.sin(x_train) + np.random.randn(N_TRAIN, 1) * SIGMA_NOISE

x_test = np.random.uniform(low=-np.pi, high=np.pi, size=N_TEST)[:, None]
y_test = np.sin(x_test) + np.random.randn(N_TEST, 1) * SIGMA_NOISE

# Initialize the neural network
model = NPModel([1, 8, 8, 1])

# Training hyperparameters
epochs = 5000
initial_lr = 0.2
lr_decay = 0.999

# Loss function
mse_loss = NPMSELoss()

# Training loop
train_losses = []
test_losses = []
lr = initial_lr
for epoch in range(epochs):
    # Forward pass
    preds = model.forward(x_train)
    loss = mse_loss.forward(preds, y_train)

    # Backward pass
    model.zero_grad()
    grad_loss = mse_loss.backward()
    model.backward(grad_loss)

    # Update weights
    model.gd_update(lr)

    # Decay learning rate
    lr *= lr_decay

    # Record training loss
    train_losses.append(loss)

    # Test loss
    test_preds = model.forward(x_test)
    test_loss = mse_loss.forward(test_preds, y_test)
    test_losses.append(test_loss)

    # Plot periodically
    if epoch % 1000 == 0 or epoch == epochs - 1:
        print("train loss", loss)
        print("test loss", test_loss)
        plt.scatter(x_train, y_train, color="blue")
        plt.plot(np.sort(x_test, axis=0), model.forward(np.sort(x_test, axis=0)), color="red")
        plt.title(f"Epoch {epoch}")
        plt.show()

# Plot loss
plt.figure()
plt.semilogy(range(epochs), train_losses, label="Train Loss")
plt.semilogy(range(epochs), test_losses, label="Test Loss")
plt.legend()
plt.show()
