import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
data = pd.read_csv(url)

# Drop missing values if any
data = data.dropna()

# Training data (first 500 rows)
train_input = np.array(data['x'][:500]).reshape(-1, 1)
train_output = np.array(data['y'][:500]).reshape(-1, 1)

# Testing data (remaining rows)
test_input = np.array(data['x'][500:]).reshape(-1, 1)
test_output = np.array(data['y'][500:]).reshape(-1, 1)

# Initialize parameters (slope `m` and intercept `c`)
m = np.random.randn(1)
c = np.random.randn(1)

# Hyperparameters
learning_rate = 0.0001
iterations = 1000

# Gradient Descent
def gradient_descent(x, y, m, c, learning_rate, iterations):
    n = len(x)
    for i in range(iterations):
        y_pred = m * x + c  # Prediction
        mse = (1/n) * np.sum((y - y_pred) ** 2)  # Mean Squared Error
        
        # Compute gradients
        dm = (-2/n) * np.sum(x * (y - y_pred))  # Gradient for `m`
        dc = (-2/n) * np.sum(y - y_pred)        # Gradient for `c`
        
        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: m = {m}, c = {c}, MSE = {mse}")
    
    return m, c

# Train the model
m, c = gradient_descent(train_input, train_output, m, c, learning_rate, iterations)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(train_input, train_output, color='blue', label='Training Data')
plt.plot(train_input, m * train_input + c, color='red', label='Regression Line')
plt.title('Linear Regression Using Gradient Descent')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Predict the output for test data
test_pred = m * test_input + c

# Calculate Mean Squared Error on test data
mse_test = (1/len(test_input)) * np.sum((test_output - test_pred) ** 2)
print(f"Test MSE: {mse_test}")