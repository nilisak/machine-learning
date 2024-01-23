import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

np.random.seed(42)


def fit_poly(x_train, y_train, degree):
    X = np.vander(x_train, degree + 1, increasing=True)
    beta = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y_train
    return beta.reshape(1, -1)


def poly(x, W):
    y = np.polyval(W[0, ::-1], x)
    return y


def mse_poly(x, y, W):
    y_pred = np.polyval(W[0, ::-1], x)
    mse = np.mean((y - y_pred) ** 2)
    return mse


# sine function
x = np.linspace(0, 2 * np.pi, 1500)
y = np.sin(x)

# training points
num_points_train = 15
x_train = np.random.uniform(0, 2 * np.pi, num_points_train)
y_train = np.sin(x_train) + np.random.normal(0, 0.1, num_points_train)

# Test points
num_points_test = 10
x_test = np.random.uniform(0, 2 * np.pi, num_points_test)
y_test = np.sin(x_test) + np.random.normal(0, 0.1, num_points_test)

# fit third order

betas = fit_poly(x_train, y_train, 3)


# get third order y

y_pol = poly(x, betas)


# get mse

mse = mse_poly(x_test, y_test, betas)


# Plotting
plt.plot(x, y, label="sin(x)")
plt.scatter(x_train, y_train, color="red", label="Training data")
plt.scatter(x_test, y_test, color="green", label="Testing data")
plt.plot(x, y_pol, label=f"Third Order Polynomial\nMSE: {mse:.4f}")
plt.title("Sine Function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()


# --------------over fitting---------------


x = np.linspace(0, 4 * np.pi, 1500)
y = np.sin(x)

# training points
num_points_train = 15
x_train = np.random.uniform(0, 4 * np.pi, num_points_train)
y_train = np.sin(x_train) + np.random.normal(0, 0.1, num_points_train)

# Test points
num_points_test = 10
x_test = np.random.uniform(0, 4 * np.pi, num_points_test)
y_test = np.sin(x_test) + np.random.normal(0, 0.1, num_points_test)

pol_degree = 15
mses = np.zeros(pol_degree)
degree = np.zeros(pol_degree)
for k in range(1, pol_degree + 1):
    betas = fit_poly(x_train, y_train, k)
    mses[k - 1] = mse_poly(x_test, y_test, betas)
    degree[k - 1] = k

# fit seventh order
betas = fit_poly(x_train, y_train, 7)
# get seventh order y
y_pol = poly(x, betas)

plt.plot(degree, mses)
plt.yscale("log")
plt.show()

plt.scatter(x_train, y_train, color="red", label="Training data")
plt.scatter(x_test, y_test, color="green", label="Testing data")
plt.plot(x, y_pol, label="seventh Order Polynomial")
plt.legend()
plt.show()


# ------------------ridge-------------------

# training points
num_points_train = 1500
x_train = np.random.uniform(0, 4 * np.pi, num_points_train)
y_train = np.sin(x_train) + np.random.normal(0, 0.1, num_points_train)

# Test points
num_points_test = 1000
x_test = np.random.uniform(0, 4 * np.pi, num_points_test)
y_test = np.sin(x_test) + np.random.normal(0, 0.1, num_points_test)


def ridge_fit_poly(x_train, y_train, k, lamb):
    # Create the design matrix X
    X = np.vander(x_train, k + 1, increasing=True)

    w = (
        np.transpose(y_train)
        @ X
        @ np.linalg.inv(np.transpose(X) @ X + lamb * np.identity((np.transpose(X) @ X).shape[0]))
    )

    return w.reshape(1, -1)


parameters = np.zeros((20, 20))
lambda_values = 10 ** np.linspace(-5, 0, 20)


for k in range(1, 21):
    for lamb_counter, lamb in enumerate(lambda_values):
        w = ridge_fit_poly(x_train, y_train, k, lamb)
        parameters[k - 1, lamb_counter] = mse_poly(x_test, y_test, w)

# Plot the results

min_value = np.min(parameters)
min_indices = np.unravel_index(np.argmin(parameters), parameters.shape)

print("Minimum Value:", min_value)
print("Indices of Minimum Value:", min_indices)
print(f"degree for minimum is {min_indices[0]+1}")
print(f"lambda for minimum is {lambda_values[min_indices[1]]}")

plt.imshow(
    parameters,
    cmap="viridis",
    aspect="auto",
    norm=LogNorm(),
)
plt.colorbar(label="MSE (log scale)")
plt.xticks(range(len(lambda_values)), [f"{val:.2e}" for val in lambda_values], rotation="vertical")
plt.yticks(range(20), [f"{val}" for val in range(1, 21)])
plt.xlabel("Lambda")
plt.ylabel("Polynomial Degree (k)")
plt.title("MSE vs Polynomial Degree and Lambda")
plt.show()


# ----------------------------cross validation ----------------------


def perform_cv(x, y, k, lamb, folds):
    # Ensure the number of folds is a divisor of N
    N = len(x)
    if N % folds != 0:
        raise ValueError("The number of folds must be a divisor of the data size N.")

    fold_size = N // folds
    mse_scores = []

    for i in range(folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size

        x_val = x[start_idx:end_idx]
        y_val = y[start_idx:end_idx]

        x_train = np.concatenate([x[:start_idx], x[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])

        # Train the model on the training set
        beta = ridge_fit_poly(x_train, y_train, k, lamb)

        # Evaluate the model on the validation set
        mse = mse_poly(x_val, y_val, beta)
        mse_scores.append(mse)

    average_mse = np.mean(mse_scores)

    return average_mse


def get_all_divisors(number):
    divisors = []
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            divisors.append(i)
            if i != number // i:
                divisors.append(number // i)
    divisors.append(number)
    return np.array(divisors)


k = 10
lambda_ = lambda_values[4]
points = 120

folds = get_all_divisors(120)
folds = np.sort(folds)

result = np.zeros(((len(folds)), 100))

for i in range(100):
    x = np.random.uniform(0, 4 * np.pi, points)
    y = np.sin(x) + np.random.normal(0, 0.1, points)
    fold_counter = 0
    for fold in folds:
        result[fold_counter, i] = perform_cv(x, y, k, lambda_, fold)
        fold_counter += 1


mean_values_per_fold = np.mean(result, axis=1)
std_dev_values_per_fold = np.std(result, axis=1)
adjusted_std_dev_values_per_fold = [
    min(mean_values_per_fold[i], std_dev_values_per_fold[i])
    for i in range(len(std_dev_values_per_fold))
]


plt.errorbar(
    folds.astype(str),
    mean_values_per_fold,
    yerr=[adjusted_std_dev_values_per_fold, std_dev_values_per_fold],
    fmt="o",
    label="Mean ± Std Dev",
)
plt.xlabel("Fold size")
plt.ylabel("Mean Value and std")
plt.legend()
plt.show()
