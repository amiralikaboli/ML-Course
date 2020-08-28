import math

import numpy as np
import pandas as pd

num_iterations = 100000
learning_rate = 1


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


if __name__ == '__main__':
    data_file_path = 'datasets/creditcard.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    X = np.array(
        [[1 for _ in range(len(data[labels[0]]))]] +
        [[feature for feature in data[label]] for label in labels[1:-2]]
    ).T
    y = np.array(data[labels[-1]])

    num_data_points, num_features = X.shape
    classes = np.unique(y)

    w = np.random.randn(num_features)
    for ind in range(1, num_iterations):
        gradient_vector = np.zeros(num_features)

        for data_point, target in zip(X, y):
            prediction = sigmoid(
                np.dot(
                    w.T,
                    data_point
                )
            )

            loss_weight = 1
            if target == 1:
                loss_weight = 578  # class population ratio

            gradient_vector += loss_weight * np.dot(
                prediction - target,
                data_point
            )

        w = w - learning_rate / math.sqrt(ind) * gradient_vector

    print(w)
