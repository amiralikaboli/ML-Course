import math

import numpy as np
import pandas as pd
from numpy import linalg

polynomial_degree = 2
regularization_factor = 1


def next_permutation(permutation):
    """
    Generates the lexicographically next permutation.

    Input: a permutation, called "a". This method modifies
    "a" in place. Returns True if we could generate a next
    permutation. Returns False if it was the last permutation
    lexicographically.
    """
    i = len(permutation) - 2
    while not (i < 0 or permutation[i] < permutation[i + 1]):
        i -= 1
    if i < 0:
        return None
    # else
    j = len(permutation) - 1
    while not (permutation[j] > permutation[i]):
        j -= 1
    permutation[i], permutation[j] = permutation[j], permutation[i]  # swap
    permutation[i + 1:] = reversed(
        permutation[i + 1:])  # reverse elements from position i+1 till the end of the sequence
    return permutation


def calc_powers_from_permutation(permutation):
    powers = []
    last_pow = 0
    for cordinate in permutation:
        if cordinate == 0:
            last_pow += 1
        else:
            powers.append(last_pow)
            last_pow = 0
    powers.append(last_pow)

    return powers


def calc_feature_from_permutation(x, permutation):
    powers = calc_powers_from_permutation(permutation)
    return math.prod(np.power(x, powers))


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

    feature_vectors = [[] for x in X]

    permutation = [0 for _ in range(polynomial_degree)] + [1 for _ in range(num_features - 1)]
    while permutation:
        for ind, x in enumerate(X):
            feature_vectors[ind].append(calc_feature_from_permutation(x, permutation))

        permutation = next_permutation(permutation)

    feature_vectors = np.array(feature_vectors)
    num_data_points, num_features = feature_vectors.shape

    best_w = np.dot(
        np.linalg.inv(
            np.add(
                np.dot(
                    feature_vectors.T,
                    feature_vectors
                ),
                regularization_factor * np.identity(num_features)
            )
        ),
        np.dot(
            feature_vectors.T,
            y
        )
    )

    print(best_w)
    print(
        np.dot(
            y - np.dot(feature_vectors, best_w),
            (y - np.dot(feature_vectors, best_w)).T
        ) / num_data_points
    )
