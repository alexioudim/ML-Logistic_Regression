from sklearn import model_selection
from sklearn import metrics
import numpy as np
import logistic_regression as lr
import generate_dataset as gd

C = np.array([[0,8],[0,8]])

C1 = np.array([[0,5],[0,5]])

C2 = np.array([[0,3],[0,3]])

# Δημιουργία του συνόλου δεδομένων
N = gd.generate_binary_problem(centers=C, N=1000) 

# Διάσπαση του συνόλου δεδομένων
x_train, x_test, y_train, y_test = model_selection.train_test_split(N[0], N[1], test_size=0.3)

# Δημιουργία του μοντέλου
model = lr.LogisticRegressionE34()

# Εκπαίδευση
model.fit(x_train, y_train, show_line=True, batch_size=None)

y_pred = model.predict(x_test) >= 0.5
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
