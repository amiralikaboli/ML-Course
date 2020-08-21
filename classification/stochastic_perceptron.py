import numpy as np
import pandas as pd

num_iterations = 1000000
learning_rate = 0.01

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
        rand_index = np.random.randint(0, num_data_points)

        prediction = np.sign(
            np.dot(
                w.T,
                X[rand_index]
            )
        )

        if y[rand_index] != prediction:
            w = w + learning_rate * np.dot(X[rand_index], y[rand_index])

    print(w)
