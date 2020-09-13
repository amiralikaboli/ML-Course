import numpy as np
import pandas as pd


def node_entropy(samples):
    if len(samples) == 0:
        return 0

    entropy = 0
    for unique_y in unique_ys:
        probability = len(np.flatnonzero(y[samples] == unique_y)) / len(samples)
        if 0 < probability < 1:
            entropy += -probability * np.log2(probability)
    return entropy


if __name__ == '__main__':
    data_file_path = 'datasets/mushrooms.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    mapped_data = []
    for label in labels:
        str_to_int = {}
        feature = []
        for feature_value in data[label]:
            if feature_value not in str_to_int:
                str_to_int[feature_value] = len(str_to_int)
            feature.append(str_to_int[feature_value])
        mapped_data.append(feature)

    X = np.array(mapped_data[1:]).T
    y = np.array(mapped_data[0])

    unique_features = [np.unique(feature) for feature in X.T]
    unique_ys = np.unique(y)

    num_data_points, num_features = X.shape

    nodes = {
        0: [ind for ind in range(num_data_points)]
    }
    leaves = [0, ]
    edges = {}
    decision_nodes = {}

    while leaves:
        leaf = leaves[0]
        leaves = leaves[1:]

        root_entropy = node_entropy(nodes[leaf])

        feature_entropys = []
        for ind, feature in enumerate(X.T):
            entropy = 0
            for unique_value in unique_features[ind]:
                samples = np.flatnonzero(feature[nodes[leaf]] == unique_value)

                child_entropy = node_entropy(samples)

                probability = len(samples) / len(nodes[leaf])
                entropy += probability * child_entropy

            feature_entropys.append(entropy)

        feature_index = int(np.argmin(feature_entropys))
        for unique_value in unique_features[feature_index]:
            samples = np.flatnonzero(X.T[feature_index][nodes[leaf]] == unique_value)
            if len(samples) == 0:
                continue

            child_entropy = node_entropy(samples)

            node_index = len(nodes)
            nodes[node_index] = samples

            if leaf not in edges:
                edges[leaf] = []
            edges[leaf].append((feature_index, unique_value, node_index))

            if child_entropy == 0:
                decision_nodes[node_index] = y[samples[0]]
            else:
                leaves.append(node_index)
