import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("BankNote_Authentication.csv")
features = df.drop(columns=["class"])
target = df["class"]


def run_experiment(training_size, verbose=False):
    no_trials = 5
    sum_accuracy = 0
    min_accuracy = 200
    max_accuracy = -1
    min_size = np.inf
    max_size = -1
    sum_size = 0
    for i in range(0, no_trials):
        training_features, test_features, training_targets, test_targets = (
            train_test_split(features, target, train_size=training_size, random_state=i)
        )

        tree = DecisionTreeClassifier().fit(training_features, training_targets)
        size = tree.tree_.node_count
        accuracy = tree.score(test_features, test_targets) * 100
        sum_accuracy += accuracy
        min_accuracy = min(min_accuracy, accuracy)
        max_accuracy = max(max_accuracy, accuracy)
        sum_size += size
        min_size = min(min_size, size)
        max_size = max(max_size, size)
        if verbose:
            print(f"{i + 1}. Tree size: {size}, accuracy: {accuracy}%")
    return (
        sum_accuracy / no_trials,
        min_accuracy,
        max_accuracy,
        sum_size / no_trials,
        min_size,
        max_size,
    )


print("Training dataset: 25%")
run_experiment(0.25, True)
print()

train_sizes = [30, 40, 50, 60, 70]
mean_accuracies = []
mean_sizes = []
for train_size in train_sizes:
    print(f"Training dataset: {train_size}%")
    mean_accuracy, min_accuracy, max_accuracy, mean_size, min_size, max_size = (
        run_experiment(train_size / 100)
    )
    mean_accuracies.append(mean_accuracy)
    mean_sizes.append(mean_size)
    print(f"Mean accuracy: {mean_accuracy}%")
    print(f"Maximum accuracy: {max_accuracy}%")
    print(f"Minimum accuracy: {min_accuracy}%")
    print(f"Mean size: {mean_size}")
    print(f"Maximum size: {max_size}")
    print(f"Minimum size: {min_size}")
    print()

plt.subplot(1, 2, 1)
plt.plot(train_sizes, mean_accuracies)
plt.xlabel("Training set size (% of the dataset)")
plt.ylabel("Mean accuracy (%)")

plt.subplot(1, 2, 2)
plt.plot(train_sizes, mean_sizes)
plt.xlabel("Training set size (% of the dataset)")
plt.ylabel("Mean size")
plt.show()
