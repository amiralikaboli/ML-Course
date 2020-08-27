import numpy as np
import pandas as pd

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

    means = {}
    covariances = {}
    for class_label in classes:
        data_point_indexes = np.flatnonzero(y == class_label)
        print(len(data_point_indexes))
        means[class_label] = sum(X[data_point_indexes]) / len(data_point_indexes)
        covariances[class_label] = np.cov(X[data_point_indexes].T)

    sum_covariance = sum(covariances.values())

    best_w = np.dot(
        np.linalg.inv(sum_covariance),
        means[classes[0]] - means[classes[1]]
    )
    best_c = np.dot(
        best_w,
        (means[classes[0]] + means[classes[1]]) / 2
    )

    print(best_w)
    print(best_c)
