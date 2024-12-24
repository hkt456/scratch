import numpy as np

import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate data
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

# Plot the data
plt.plot(x, y, label='Sigmoid Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.title('Sigmoid Function Illustration')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('graphs/sigmoid_function.png')
plt.show()