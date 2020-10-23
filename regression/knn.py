import numpy as np
import pandas as pd

k = 10

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

    sse_error = 0
    for validation_data_point in validation_data_points:
        distance_with_training_data_points = []
        for training_data_point in training_data_points:
            distance_with_training_data_points.append(
                (np.linalg.norm(training_data_point[:-1] - validation_data_point[:-1]), training_data_point[-1])
            )

        distance_with_training_data_points = sorted(distance_with_training_data_points)
        training_data_points_value = [data_point[1] for data_point in distance_with_training_data_points][:k]

        mean_nearest_values = np.mean(training_data_points_value)

        diff_value = abs(validation_data_point[-1] - mean_nearest_values)
        sse_error += diff_value ** 2

    print(sse_error)
