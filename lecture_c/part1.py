import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# Generate training data
N_TRAIN = 15
SIGMA_NOISE = 0.1
torch.manual_seed(0xDEADBEEF)
x_train = torch.rand(N_TRAIN) * 2 * torch.pi
y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE


# Implement closed-form solution
def fit_poly_closed_form(x_train, y_train, degree):
    X = torch.vander(x_train, degree + 1, increasing=True)
    beta = torch.linalg.inv(X.T @ X) @ X.T @ y_train
    return beta


degree = 3
W_closed_form = fit_poly_closed_form(x_train, y_train, degree)


# Define the polynomial regression model
class PolynomialModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialModel, self).__init__()
        self.weights = nn.Parameter(torch.ones(degree + 1))

    def forward(self, x):
        # Create polynomial features and apply weights
        poly_features = torch.vander(x, len(self.weights), increasing=True)
        return poly_features @ self.weights


# Initialize weights for the polynomial model
def initialize_weights(model):
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            if i == 0:  # Weights initialization
                param[:] = torch.tensor([1 / (n + 1) for n in range(len(param))])
            else:  # Bias initialization (if any)
                param.zero_()


# Function to train the model
def train_polynomial_model(model, optimizer, num_steps=100):
    criterion = nn.MSELoss()
    loss_history = []
    for step in range(num_steps):
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history


# Experiment with different learning rates
learning_rates = [0.0000001, 0.000001, 0.00001]
loss_histories = {}
for lr in learning_rates:
    model = PolynomialModel(degree)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = train_polynomial_model(model, optimizer)
    loss_histories[lr] = loss_history

# Plot loss vs. training steps for different learning rates
plt.figure(figsize=(12, 6))
for lr, loss_history in loss_histories.items():
    plt.plot(loss_history, label=f"LR={lr}")
# plt.yscale("log")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss vs. Training Steps for Different Learning Rates")
plt.legend()
plt.show()

# Plot comparison with ground truth, training data, exact solution, and optimized polynomial
plt.figure(figsize=(12, 6))
x_values = torch.linspace(0, 2 * torch.pi, 500)
plt.plot(x_values, torch.sin(x_values), label="Ground Truth (sin(x))")
plt.scatter(x_train, y_train, color="red", label="Training Data")

# Plot exact solution from closed-form method
y_exact = torch.vander(x_values, degree + 1, increasing=True) @ W_closed_form
plt.plot(x_values, y_exact, label="Exact Solution (Closed-Form)", color="green")

# Plot optimized polynomial with the best learning rate
best_lr = min(loss_histories, key=lambda lr: loss_histories[lr][-1])
best_model = PolynomialModel(degree)
optimizer = optim.SGD(best_model.parameters(), lr=best_lr)
train_polynomial_model(best_model, optimizer, num_steps=100)
y_optimized = best_model(x_values).detach()
plt.plot(x_values, y_optimized, label=f"Optimized Polynomial (LR={best_lr})", color="purple")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Polynomial Regression Solutions")
plt.legend()
plt.show()


# Train using SGD with Momentum
final_losses = {}

# Iterate over combinations of learning rate and momentum
learning_rates = 10 ** torch.linspace(-6, 0, 7)
momentums = torch.linspace(0.1, 10, 20)
for lr in learning_rates:
    for momentum in momentums:
        # Initialize model and optimizer with current lr and momentum
        model_sgd_momentum = PolynomialModel(degree)
        initialize_weights(model_sgd_momentum)
        optimizer_sgd_momentum = optim.SGD(
            model_sgd_momentum.parameters(), lr=lr.item(), momentum=momentum.item()
        )

        # Train the model
        loss_history_sgd_momentum = train_polynomial_model(
            model_sgd_momentum, optimizer_sgd_momentum
        )

        # Store the final loss with the corresponding lr and momentum
        final_losses[(lr.item(), momentum.item())] = loss_history_sgd_momentum[-1]

# Find the combination with the minimum final loss
best_lr, best_momentum = min(final_losses, key=final_losses.get)
best_loss = final_losses[(best_lr, best_momentum)]

print(f"Best Learning Rate SGD momentum: {best_lr}, Best Momentum SGD momentum: {best_momentum}")
print(f"Minimum Final Loss SGD momentum: {best_loss:.4f}")

model_sgd_momentum = PolynomialModel(degree)
initialize_weights(model_sgd_momentum)
optimizer_sgd_momentum = optim.SGD(
    model_sgd_momentum.parameters(), lr=best_lr, momentum=best_momentum
)
loss_history_sgd_momentum = train_polynomial_model(model_sgd_momentum, optimizer_sgd_momentum)
print(f"Final Loss after 100 steps (SGD momentum): {loss_history_sgd_momentum[-1]:.4f}")


# Train using Adam optimizer
final_losses = defaultdict(float)

# Iterate over learning rates
for learn_rate in 10 ** torch.linspace(-5, 0, 6):
    model_adam = PolynomialModel(degree)
    initialize_weights(model_adam)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=learn_rate)
    loss_history_adam = train_polynomial_model(model_adam, optimizer_adam)
    final_losses[learn_rate.item()] = loss_history_adam[-1]

# Find the learning rate with the minimum final loss
best_learn_rate = min(final_losses, key=final_losses.get)
best_loss = final_losses[best_learn_rate]

print("Best learning rate Adam:", best_learn_rate)
print("Minimum final loss Adam:", best_loss)

model_adam = PolynomialModel(degree)
initialize_weights(model_adam)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_learn_rate)
loss_history_adam = train_polynomial_model(model_adam, optimizer_adam)
print(f"Final Loss after 100 steps (Adam): {loss_history_adam[-1]:.4f}")


# Train using LBFGS optimizer
def closure():
    criterion = nn.MSELoss()
    optimizer_lbfgs.zero_grad()
    predictions = model_lbfgs(x_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    return loss


model_lbfgs = PolynomialModel(degree)
initialize_weights(model_lbfgs)
optimizer_lbfgs = optim.LBFGS(model_lbfgs.parameters(), lr=1)
loss_history_lbfgs = []

for step in range(100):
    optimizer_lbfgs.step(closure)
    loss_history_lbfgs.append(closure().item())
print(f"Final Loss after 100 steps (LBFGS): {loss_history_lbfgs[-1]:.4f}")

# Plot loss against steps for SGD with Momentum, Adam, and LBFGS
plt.figure(figsize=(12, 6))
plt.plot(loss_history_sgd_momentum, label="SGD with Momentum")
plt.plot(loss_history_adam, label="Adam")
plt.plot(loss_history_lbfgs, label="LBFGS")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss vs. Training Steps for Different Optimizers")
plt.legend()
plt.show()


# Function to plot the fitted polynomial
def plot_fitted_polynomial(model, label):
    y_pred = model(x_values).detach()
    plt.plot(x_values, y_pred, label=label)


# Plot the final optimized polynomial for each optimizer
plt.figure(figsize=(12, 6))
plt.scatter(x_train, y_train, color="red", label="Training Data")
plt.plot(x_values, torch.sin(x_values), label="Ground Truth (sin(x))", color="blue")
plot_fitted_polynomial(model_sgd_momentum, "SGD with Momentum")
plot_fitted_polynomial(model_adam, "Adam")
plot_fitted_polynomial(model_lbfgs, "LBFGS")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Polynomial Regression Fits with Different Optimizers")
plt.legend()
plt.show()
