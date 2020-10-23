import numpy as np
import pandas as pd

k = 50

if __name__ == '__main__':
    data_file_path = 'datasets/winequality-red.csv'
    data = pd.read_csv(data_file_path)

    labels = list(data.keys())
    all_data_points = np.array(
        [[feature for feature in data[label]] for label in labels]
    ).T

    num_all_data_points, num_features = all_data_points.shape

    np.random.shuffle(all_data_points)

    num_training_data_points = int(num_all_data_points * 0.8)
    training_data_points = all_data_points[:num_training_data_points]
    validation_data_points = all_data_points[num_training_data_points:]

    num_misclassified_data_points = 0
    for validation_data_point in validation_data_points:
        distance_with_training_data_points = []
        for training_data_point in training_data_points:
            distance_with_training_data_points.append(
                (np.linalg.norm(training_data_point[:-1] - validation_data_point[:-1]), training_data_point[-1])
            )

        distance_with_training_data_points = sorted(distance_with_training_data_points)
        training_data_points_class = [data_point[1] for data_point in distance_with_training_data_points][:k]

        most_frequent_class = max(training_data_points_class, key=training_data_points_class.count)

        if most_frequent_class != validation_data_point[-1]:
            num_misclassified_data_points += 1

    print(1 - num_misclassified_data_points / len(validation_data_points))
