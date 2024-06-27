import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 20000, 1000)
y = np.sin(x / 1000)

# Create a plot
fig, ax = plt.subplots()

# Plot data
ax.plot(x, y)

# Set the x-axis limit to range from 0 to 20000
ax.set_xlim(0, 20000)

# Optionally, set the y-axis limit if needed
# ax.set_ylim(-1, 1)

# Add labels and title
ax.set_xlabel('X-axis (0 to 20000)')
ax.set_ylabel('Y-axis (sin(x/1000))')
ax.set_title('X-axis from 0 to 20000')

# Show plot
plt.show()
