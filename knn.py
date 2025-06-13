import numpy as np

class KNN:
    def __init__(self , k):
        self.k = k
    
    #training
    def fit(self ,X ,Y):
        self.X_train = X
        self.Y_train = Y
        
    def euclidean_distance(self, x1 ,x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self , X):
        Y = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                d = self.euclidean_distance(x ,x_train)
                distances.append(d)   # distances = [1.21165165 , 3.24298468 , 2.268298424 , 1.284642426 , 0.28249498 , ...]
            
            clean_distances = [p.item() for p in distances]
            nearest_neighbors = np.argsort(clean_distances)[0:self.k] # if k=3 --> nearest_neighbors = (121 , 34 , 12)andis array
            
            result = np.bincount(self.Y_train[nearest_neighbors]) # result =(2,1) --> zero andis = 2   , one andis = 1
            y = np.argmax(result) # 2 > 1 --> we have 2 andis of zero --> y = 0
            Y.append(y)
        return Y
    
    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.sum(Y_pred == Y)/len(Y)
        return float(accuracy)
        
        
        
        
