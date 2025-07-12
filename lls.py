import numpy as np
from numpy.linalg import inv




# w = شیب خط
# w = (X.T * X) ** -1 * X.T * Y
# w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), X_train.T), Y_train)
class LLS:
    def __init__(self):
        self.w = None
    
    def fit(self, X_train, Y_train):
        #self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), X_train.T), Y_train)
        self.w = inv(X_train.T @ X_train) @ X_train.T @ Y_train
        return self.w
        
    def predict(self, X_test):
        Y_pred = X_test @ self.w
        return Y_pred
    
    def evall
    
    