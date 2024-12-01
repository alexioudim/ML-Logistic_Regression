from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np

# Load the dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

for _ in range(20):
    # Split the dataset
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    # Normalize the data
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create the model
    model = linear_model.LogisticRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Test the model's effectiveness
    y_pred = model.predict(x_test) >= 0.5
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print (f"Accuracy: {accuracy}")

