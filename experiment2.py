from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import logistic_regression as lr
import time
import platform
import psutil

# Load the dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target
accuracies = []

start_time = time.time()
for _ in range(20):
    # Split the dataset
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    # Normalize the data
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create the model
    model = lr.LogisticRegressionE34()

    # Train the model
    model.fit(x_train, y_train, batch_size=64)

    # Test the model's effectiveness
    y_pred = model.predict(x_test) >= 0.5
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print (f"Accuracy: {accuracy}")

end_time = time.time()

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

execution_time = end_time - start_time

# Print results
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}")
print(f"Execution time: {execution_time}")