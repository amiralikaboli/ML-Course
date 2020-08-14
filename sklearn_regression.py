import random

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, preprocessing, pipeline

polynomial_degree = 2

if __name__ == '__main__':
    data_file_path = 'datasets/winequality-red.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    X = np.array(
        [[feature for feature in data[label]] for label in labels[:-1]]
    ).T
    y = np.array(data[labels[-1]])

    num_data_points, num_features = X.shape
    random.shuffle(X)

    regressor = pipeline.make_pipeline(
        preprocessing.PolynomialFeatures(polynomial_degree),
        linear_model.LinearRegression()
    )
    regressor.fit(X, y)

    preds = regressor.predict(X)

    print(metrics.mean_squared_error(y, preds))
