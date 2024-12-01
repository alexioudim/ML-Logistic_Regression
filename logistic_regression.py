import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionE34:
    def __init__(self, lr=0.01): # 10**-2
        self.w = None  # Βάρη του μοντέλου
        self.b = None  # Όρος μεροληψίας
        self.lr = lr  # Ρυθμός εκμάθησης
        self.N = None  # Πλήθος των δεδομένων εκπαίδευσης
        self.p = None  # Διάσταση του χώρου των χαρακτηριστικών
        self.l_grad_w = None  # Παράγωγοι των w σε κάθε βήμα βελτιστοποίησης
        self.l_grad_b = None  # Παράγωγοι του b σε κάθε βήμα βελτιστοποίησης
        self.prediction = None  # Πρόβλεψη p(1|x) για τα δεδομένα εισόδου x κατά την εκπαίδευση
        self.f = None # Αποθήκευση του αποτελέσματος του υπολογισμού pmodel

    def init_parameters(self):
        self.w = np.random.randn(self.p) * 0.1 # Δημιουργία διανύσματος βαρών με μήκος p
        self.b = np.random.randn() * 0.1 #Δημιουργία όρου μεροληψίας

    def forward(self, X):
        z = X @ self.w + self.b # Υπολογισμός διανύσματος
        self.prediction = 1 / (1 + np.exp(-z))
        self.f = self.prediction

    def predict(self, X):
        z = X @ self.w + self.b
        self.prediction = 1 / (1 + np.exp(-z))
        return self.prediction
    
    def loss(self, X, y):
        self.predict(X)
        self.N = X.shape[0] # Υπολογίζω το πλήθος δεδομένων 
        loss = -(1 / self.N) * np.sum(y * np.log(self.prediction) + (1 - y) * np.log(1 - self.prediction))
        return loss

    def backward(self,X, y):
        self.predict(X)
        self.N = X.shape[0]
        self.l_grad_w = -(1 / self.N) * np.dot((y - self.prediction), X) 
        self.l_grad_b = -(1 / self.N) * np.sum(y - self.prediction)
        return self.l_grad_w, self.l_grad_b
    
    def step(self):
        self.w -= self.lr * self.l_grad_w
        self.b -= self.lr * self.l_grad_b

    def show_line(self, X: np.ndarray, y: np.ndarray) -> None:

        if (X.shape[1] != 2):
            print("Not plotting: Data is not 2-dimensional")
            return
        idx0 = (y == 0)
        3
        idx1 = (y == 1)
        X0 = X[idx0, :2]
        X1 = X[idx1, :2]
        plt.plot(X0[:, 0], X0[:, 1], 'gx')
        plt.plot(X1[:, 0], X1[:, 1], 'ro')
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
        yline = (self.w[0]*xline + self.b) / (-self.w[1])
        plt.plot(xline, yline, 'b')
        plt.show()

    def fit(self, X, y, iterations=10000, batch_size=None,
            show_step=1000, show_line=False):
        
        # Έλεγχοι
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Τα δεδομένα δεν είναι numpy arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Τα δεδομένα δεν έχουν συμβατές διαστάσεις")
        
        # Αρχικοποίηση παραμέτρων
        self.p = X.shape[1]
        self.init_parameters()

        # Τοποθέτηση δειγμάτων σε τυχαία σειρά
        values = np.arange(X.shape[0])
        np.random.shuffle(values)
        X = X[values]
        y = y[values]

        # Επανάληψη εκπαίδευσης
        for iteration in range(iterations):
            if batch_size is None:
                batch_X = X
                batch_y = y
            else:
                start = (iteration * batch_size) % X.shape[0]
                end = start + batch_size
                batch_X = X[start:end]
                batch_y = y[start:end]

            # Εκπαίδευση
            self.forward(batch_X)

            self.backward(batch_X, batch_y)

            self.step()

            # Απώλεια
            if (iteration + 1) % show_step == 0:
                current_loss = self.loss(X,y)
                print(f"Iteration {iteration + 1}: Loss = {current_loss}")
                if show_line:
                    self.show_line(X,y)
    
    