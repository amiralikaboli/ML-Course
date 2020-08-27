import numpy as np
import pandas as pd

num_iterations = 1000
learning_rate = 0.01


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


if __name__ == '__main__':
    data_file_path = 'datasets/creditcard.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    X = np.array(
        [[feature for feature in data[label]] for label in labels[1:-2]]
    ).T
    y = np.array(data[labels[-1]])

    num_data_points, num_features = X.shape
    classes = np.unique(y)

    w = np.random.randn(num_features)
    for _ in range(num_iterations):
        gradient_vector = np.zeros(num_features)

        for data_point, target in zip(X, y):
            gradient_vector += np.dot(
                sigmoid(
                    np.dot(
                        w.T,
                        data_point
                    )
                ) - target,
                data_point
            )

        w = w + learning_rate / num_data_points * gradient_vector

    print(w)
