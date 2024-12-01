from sklearn import model_selection
import numpy as np
import logistic_regression as lr
import generate_dataset as gd

C = np.array([[0,8],[0,8]])

print(C)

N = gd.generate_binary_problem(centers=C, N=1000) 

x_train, x_test, y_train, y_test = model_selection.train_test_split(N[0], N[1], test_size=0.3)

model = lr.LogisticRegressionE34()

model.fit(x_train, y_train, show_line=True, batch_size=None)