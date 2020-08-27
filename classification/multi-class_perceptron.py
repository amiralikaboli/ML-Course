import numpy as np
import pandas as pd

num_iterations = 1000000
learning_rate = 0.01

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
    classes = np.unique(y)

    w = {class_label: np.random.randn(num_features) for class_label in classes}
    for _ in range(num_iterations):
        rand_index = np.random.randint(0, num_data_points)

        predictions = [
            (
                np.sign(
                    np.dot(
                        w[class_label].T,
                        X[rand_index]
                    )
                ),
                class_label
            )
            for class_label in classes
        ]

        predicted_class = sorted(predictions)[-1][1]

        if y[rand_index] != predicted_class:
            w[predicted_class] = w[predicted_class] - learning_rate * X[rand_index]
            w[y[rand_index]] = w[y[rand_index]] + learning_rate * X[rand_index]

    print(w)
