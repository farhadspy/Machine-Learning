import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from knn import KNN

        

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    Y = iris.target
    
    X_train , X_test , Y_train  , Y_test = train_test_split(X,Y,test_size=0.2)
    knn = KNN(k=3)
    knn. fit(X_train,Y_train)
    accuracy = knn.evaluate(X_test,Y_test)
    print("accuracy: ", accuracy)
    
    knn_sklearn = KNeighborsClassifier(n_neighbors= 3)
    knn_sklearn. fit(X_train,Y_train)
    accuracy = knn_sklearn.score(X_test,Y_test)
    print("accuracy: ", accuracy)