from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import logistic_regression as lr
import time

# Φόρτωση του συνόλου δεδομένων
data = datasets.load_breast_cancer()
X, y = data.data, data.target
accuracies = []

# Έναρξη χρονομέτρησης      
start_time = time.time()
for _ in range(20):
    # Διάσπαση του συνόλου δεδομένων
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

    # Κανονικοποίηση των δεδομένων
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Δημιουργία του μοντέλου
    model = lr.LogisticRegressionE34()

    # Εκπαίδευση    
    model.fit(x_train, y_train, batch_size=64)

    # Αξιολόγηση της ευστοχίας
    y_pred = model.predict(x_test) >= 0.5
    accuracy = metrics.precision_score(y_test, y_pred)
    accuracies.append(accuracy)
    print (f"Accuracy: {accuracy}")

# Τέλος χρονομέτρησης
end_time = time.time()

# Υπολογισμός
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
execution_time = end_time - start_time

# Εμφάνιση αποτελεσμάτων
print(f"Mean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}")
print(f"Execution time: {execution_time}")