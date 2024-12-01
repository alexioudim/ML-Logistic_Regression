from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import logistic_regression as lr

# Load the dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target
mean_accuracy = 0

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
    mean_accuracy += accuracy
    print (f"Accuracy: {accuracy}")

print(f"Mean Accuracy: {mean_accuracy/20}")

