import numpy as np
import matplotlib.pyplot as plt

# Dataset
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([25, 30, 42, 48, 50, 65, 70, 75, 90, 105])

# Function for Locally Weighted Regression
def predict(point):
    weights = np.exp(-((x - point) ** 2) / 8)

    # Weighted averages
    mean_x = np.sum(weights * x) / np.sum(weights)
    mean_y = np.sum(weights * y) / np.sum(weights)

    # Calculate slope
    numerator = np.sum(weights * (x - mean_x) * (y - mean_y))
    denominator = np.sum(weights * (x - mean_x) ** 2)

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    return slope * point + intercept

# Predict values at many points
predicted = []

for point in x:
    predicted.append(predict(point))

# Draw graph
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, predicted, color="red", label="LWR Fitted Curve")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Locally Weighted Regression")
plt.legend()
plt.grid()
plt.show()
