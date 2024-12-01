from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np
import time

# Φόρτωση του συνόλου δεδομένων
data = datasets.load_breast_cancer()
X, y = data.data, data.target
precisions = []

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
    model = linear_model.LogisticRegression()

    # Εκπαίδευση    
    model.fit(x_train, y_train)

    # Αξιολόγηση της ευστοχίας
    y_pred = model.predict(x_test) >= 0.5
    precision = metrics.precision_score(y_test, y_pred)
    precisions.append(precision)
    print (f"Precision: {precision}")

# Τέλος χρονομέτρησης
end_time = time.time()

# Υπολογισμός
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
execution_time = end_time - start_time

# Εμφάνιση αποτελεσμάτων
print(f"Mean precision: {mean_precision}")
print(f"Standard deviation of precision: {std_precision}")
print(f"Execution time: {execution_time}")

