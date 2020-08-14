import numpy as np
import pandas as pd

num_iterations = 10000
learning_rate = 0.00001

if __name__ == '__main__':
    data_file_path = 'datasets/winequality-red.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    X = np.array(
        [[1 for _ in range(len(data[labels[0]]))]] +
        [[feature for feature in data[label]] for label in labels[:-1]]
    ).T
    y = np.array(data[labels[-1]])

    num_data_points, num_features = X.shape

    w = np.random.randn(num_features)
    for _ in range(num_iterations):
        gradient_vector = np.dot(
            X.T,
            X.dot(w) - y
        )
        w = w - (learning_rate / num_data_points) * gradient_vector

    print(w)
    print(
        np.dot(
            y - np.dot(X, w),
            (y - np.dot(X, w)).T
        )
    )
