import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from relu import ReLUHKT  # Import the ReLU class

# Instantiate the ReLU class
relu_instance = ReLUHKT.apply

# Generate x values
x = torch.linspace(-10, 10, 400, dtype=torch.float32, requires_grad=True)


# Compute y values using the ReLU function
y = relu_instance(x)

y = y.detach().numpy()
x = x.detach().numpy()

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ReLU(x)')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.legend()
plt.grid(True)

# Ensure the directory exists
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'graphs')
os.makedirs(output_dir, exist_ok=True)

# Save the plot
output_path = os.path.join(output_dir, 'relu_function.png')
plt.savefig(output_path)
plt.close()

print(f"Graph saved to {output_path}")
