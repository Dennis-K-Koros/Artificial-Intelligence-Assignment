import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/Dennis Koros/Downloads/Nairobi Office Price Ex.csv')  # Update the path if needed

# Extract relevant columns
X = data['SIZE'].values  # Feature (office size)
y = data['PRICE'].values  # Target (office price)

# Define Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * X + c
    # Calculate gradients
    dm = (-2 / n) * np.sum(X * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)
    # Update weights
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Initialize slope (m) and intercept (c)
m, c = np.random.rand(), np.random.rand()
learning_rate = 0.0001  # Decrease the learning rate for finer updates
epochs = 1000  # Increase the number of epochs for better convergence

# Training loop
errors = []

for epoch in range(epochs):
    # Predict with current slope and intercept
    y_pred = m * X + c
    # Calculate MSE and track it
    mse = mean_squared_error(y, y_pred)
    errors.append(mse)
    # Update weights using gradient descent
    m, c = gradient_descent(X, y, m, c, learning_rate)

# Plotting the line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, m * X + c, color="red", label="Line of Best Fit")
plt.xlabel("Office Size (sq. ft.)")
plt.ylabel("Office Price")
plt.legend()
plt.title("Linear Regression - Line of Best Fit")
plt.show()

# Predict for an office size of 100 sq. ft.
predicted_price = m * 100 + c
print(f"Predicted price for a 100 sq. ft. office: {predicted_price:.2f}")
