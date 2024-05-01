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


# Separate features and targets
def get_features(data):
    return [[line[0], line[1], line[2], line[3]] for line in data]


def get_targets(data):
    return [line[4] for line in data]


train_features = get_features(train_data)
train_targets = get_targets(train_data)
test_features = get_features(test_data)
test_targets = get_targets(test_data)


# Mean, std deviation
def get_mean(idx, features):
    return sum(x[idx] for x in features) / len(features)


def get_std_dev(idx, mean, features):
    return (sum((x[idx] - mean) ** 2 for x in features) / len(features)) ** 0.5


def normalize(v, mean, std):
    return (v - mean) / std


def preprocess_features(features):
    mean_0 = get_mean(0, features)
    std_dev_0 = get_std_dev(0, mean_0, features)
    mean_1 = get_mean(1, features)
    std_dev_1 = get_std_dev(1, mean_1, features)
    mean_2 = get_mean(2, features)
    std_dev_2 = get_std_dev(2, mean_2, features)
    mean_3 = get_mean(3, features)
    std_dev_3 = get_std_dev(3, mean_3, features)
    return [
        [
            normalize(x[0], mean_0, std_dev_0),
            normalize(x[1], mean_1, std_dev_1),
            normalize(x[2], mean_2, std_dev_2),
            normalize(x[3], mean_3, std_dev_3),
        ]
        for x in features
    ]


train_features_preprocessed = preprocess_features(train_features)
