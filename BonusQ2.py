# Define the training data
X = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
     (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
y = [0, 0, 0, 0, 0, 0, 0, 1]


# Define the Perceptron function
def perceptron(x, y, learning_rate=0.1, epochs=100):
    # Initialize weights and bias to zero
    weights = [0, 0, 0]
    bias = 0

    # Train the Perceptron for a specified number of epochs
    for epoch in range(epochs):
        for i in range(len(x)):
            # Calculate the weighted sum of inputs
            z = sum([x[i][j] * weights[j] for j in range(len(weights))]) + bias
            # Apply the step function to get the predicted output
            y_pred = 1 if z > 0 else 0
            # Update the weights and bias based on the error
            weights = [weights[j] + learning_rate * (y[i] - y_pred) * x[i][j] for j in range(len(weights))]
            bias += learning_rate * (y[i] - y_pred)

    return weights, bias


if __name__ == '__main__':
    # Train the Perceptron on the provided data
    weights, bias = perceptron(X, y)

    # Test the trained Perceptron on all possible input combinations
    for x in X:
        z = sum([x[i] * weights[i] for i in range(len(weights))]) + bias
        y_pred = 1 if z > 0 else 0
        print(f"Input: {x}, Predicted Output: {y_pred}")