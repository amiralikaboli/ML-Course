import numpy as np
import pandas as pd

num_iterations = 10000
learning_rate = 0.0001

if __name__ == '__main__':
    data_file_path = 'datasets/creditcard.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    X = np.array(
        [[1 for _ in range(len(data[labels[0]]))]] +
        [[feature for feature in data[label]] for label in labels[1:-2]]
    ).T
    y = np.array([2 * zero_or_one - 1 for zero_or_one in data[labels[-1]]])  # 0, 1 to -1, 1

    num_data_points, num_features = X.shape

    w = np.random.randn(num_features)
    for _ in range(num_iterations):
        gradient_vector = np.zeros(num_features)

        for data_point, target in zip(X, y):
            prediction = np.sign(
                np.dot(
                    w.T,
                    data_point
                )
            )

            if target != prediction:
                gradient_vector += target * data_point

        w = w + learning_rate * gradient_vector

    print(w)
