import numpy as np
import pandas as pd

if __name__ == '__main__':
    data_file_path = 'datasets/USA_Housing.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    X = np.array(
        [[1 for _ in range(len(data[labels[0]]))]] +
        [[feature for feature in data[label]] for label in labels[:-1]]
    ).T
    y = np.array(data[labels[-1]])

    num_data_points, num_features = X.shape

    best_w = np.dot(
        np.linalg.inv(
            np.dot(
                X.T,
                X
            )
        ),
        np.dot(
            X.T,
            y
        )
    )

    print(best_w)
    print(
        np.dot(
            y - np.dot(X, best_w),
            (y - np.dot(X, best_w)).T
        ) / num_data_points
    )
