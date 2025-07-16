from matplotlib.pylab import astype
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import time

# Preprocessing Input Data
data = pd.read_csv("C:/Users/Farhad/Desktop/python/Machine_Learning/ANSUR_II_FEMALE_Public.csv")
data2 = pd.read_csv("C:/Users/Farhad/Desktop/python/Machine_Learning/ANSUR_II_MALE_Public.csv" , encoding='latin-1')


data = pd.concat([data, data2])


data["weightkg"] = data["weightkg"] / 10 # convert to kg
data["Heightin"] = data["Heightin"] * 2.54  # convert to cm
#data["Gender"] = data["Gender"].replace(["Female","Male"],[0,1])   # convert to binary

X = data["Heightin"].values
Y = data["weightkg"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,shuffle=True, test_size=0.99)

# Reshape the data

X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

fig, (ax1, ax2) = plt.subplots(1, 2)

# Training
w = np.random.rand(1, 1)  
print("w:", w)

b = np.random.rand(1, 1)  # Bias 

#plt.scatter(X_train, Y_train, color='blue')
#plt.show()

learning_rate_w = 0.00001
learning_rate_b = 0.1
losses = []
epochs = 20

for j in range(epochs):
    for i in range(X_train.shape[0]):
        x = X_train[i]
        y = Y_train[i]
        
        # Calculate the prediction
        y_pred = x * w + b
        
        # Calculate the error
        error = y - y_pred
        
        # SGD Update
        w = w + (error * x * learning_rate_w)
        b = b + (error * learning_rate_b)
        print("w:", w, "b:", b)
        #time.sleep(0.5)
        
        
        # mae
        #loss = np.mean(np.abs(error))
        #losses.append(loss)
        
        # mse
        loss = np.mean(error ** 2)
        losses.append(loss)
        
        
        Y_pred = X_train * w + b
        ax1.clear()
        ax1.scatter(X_train, Y_train, color='blue')
        ax1.plot(X_train, Y_pred, color='red')
        
        ax2.clear()
        ax2.plot(losses, color='green')
        plt.pause(0.1)