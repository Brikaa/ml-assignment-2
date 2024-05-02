import random

# Read the data
file = open("BankNote_Authentication.csv")
file.readline()
lines = file.readlines()
file.close()

lines = [line.split(",") for line in lines]
for i in range(len(lines)):
    lines[i] = list(map(float, [x for x in lines[i]]))

# Shuffle the data
for i in range(len(lines)):
    r = random.randrange(len(lines))
    tmp = lines[i]
    lines[i] = lines[r]
    lines[r] = tmp

# 70% training, 30% test
train_size = round(0.7 * len(lines))
train_data = lines[:train_size]
test_data = lines[train_size:]


# Mean, std deviation
def get_mean(idx, data):
    return sum(x[idx] for x in data) / len(data)


def get_std_dev(idx, mean, data):
    return (sum((x[idx] - mean) ** 2 for x in data) / len(data)) ** 0.5


def normalize(v, mean, std):
    return (v - mean) / std


def preprocess_data(data):
    mean_0 = get_mean(0, data)
    std_dev_0 = get_std_dev(0, mean_0, data)
    mean_1 = get_mean(1, data)
    std_dev_1 = get_std_dev(1, mean_1, data)
    mean_2 = get_mean(2, data)
    std_dev_2 = get_std_dev(2, mean_2, data)
    mean_3 = get_mean(3, data)
    std_dev_3 = get_std_dev(3, mean_3, data)
    return [
        [
            normalize(x[0], mean_0, std_dev_0),
            normalize(x[1], mean_1, std_dev_1),
            normalize(x[2], mean_2, std_dev_2),
            normalize(x[3], mean_3, std_dev_3),
            x[4],
        ]
        for x in data
    ]


train_data_preprocessed = preprocess_data(train_data)
test_data_preprocessed = preprocess_data(test_data)


def euclidean(f1, f2):
    return (
        (f1[0] - f2[0]) ** 2
        + (f1[1] - f2[1]) ** 2
        + (f1[2] - f2[2]) ** 2
        + (f1[3] - f2[3]) ** 2
    ) ** 0.5


# Use KNN to classify test data
def knn(k, train_data, X):
    # add one element to train_data representing the distance
    # sort based on distance
    # pick the first k
    # see which class is most common
    train_data_with_distances = [f + [euclidean(f, X)] for f in train_data]
    train_data_with_distances.sort(key=lambda x: x[-1])
    first_k = train_data_with_distances[:k]
    count_0 = sum(x[4] == 0 for x in first_k)
    count_1 = k - count_0
    if count_0 == count_1:
        return random.choice([0, 1])
    elif count_0 > count_1:
        return 0
    else:
        return 1


for k in range(1, 10):
    correctly_classified_instances = sum(
        knn(k, train_data_preprocessed, v) == v[-1] for v in test_data_preprocessed
    )
    print(f"K: {k}")
    print(f"Number of correctly classified instances: {correctly_classified_instances}")
    print(f"Total number of instances in the test set: {len(test_data_preprocessed)}")
    print(
        f"Accuracy: {correctly_classified_instances / len(test_data_preprocessed) * 100}%"
    )
    print()
