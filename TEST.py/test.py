import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap






female_data = pd.read_csv("C:/Users/Farhad/Desktop/python/Machine_Learning/ANSUR_II_FEMALE_Public.csv")
female_data.head()

male_data = pd.read_csv("C:/Users/Farhad/Desktop/python/Machine_Learning/ANSUR_II_MALE_Public.csv" , encoding='latin-1')
male_data.head()

# جدا کردن 50 ردیف اول به‌صورت دیتا فریم (بدون تبدیل به لیست)
test_female_data = female_data.head(50).copy()  # استفاده از copy برای جلوگیری از مشکلات مرجع
test_male_data = male_data.head(50).copy()

# حذف 50 ردیف اول
female_data = female_data.iloc[50:]  # همه ردیف‌ها از 50 به بعد
female_data.head()

# حذف 50 ردیف اول
male_data = male_data.iloc[50:]  # همه ردیف‌ها از 50 به بعد
male_data.head()

data = pd.concat([female_data, male_data])

data2 = pd.concat([test_female_data, test_male_data])

X_test = female_data.head(50)[['stature', 'weightkg']].values
y_test = female_data.head(50)['SubjectNumericRace'].values
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#preprocess
data["weightkg"] = data["weightkg"] / 10 # convert to kg
data["stature"] = data["stature"] / 10   # convert to cm
data["Gender"] = data["Gender"].replace(["Female","Male"],[0,1])   # convert to binary

data2["weightkg"] = data2["weightkg"] / 10 # convert to kg
data2["stature"] = data2["stature"] / 10   # convert to cm
data2["Gender"] = data2["Gender"].replace(["Female","Male"],[0,1])   # convert to binary


cmap = ["red" , "blue"]
plt.scatter(data["stature"] ,data["weightkg"] , c=data["Gender"] ,cmap=ListedColormap(cmap) , alpha=0.1)
plt.title("ansur data")
plt.xlabel('stature')
plt.ylabel('weightkg')
plt.show()

def generate_dataset():
    
    width = data["stature"]
    length = data["weightkg"]
    x = np.array((width , length)).T

    y = data["Gender"]
    
    return x , y

def generate_dataset2():
    
    width2 = data2["stature"]
    length2 = data2["weightkg"]
    x = np.array((width2 , length2)).T

    y = data["Gender"]
    
    return x , y

X_train , Y_train = generate_dataset()
X_test , Y_test = generate_dataset2()


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
        
        
knn = KNN(k = 11)
knn.fit(X_train,Y_train)


people_1 = np.array([170, 1])
people_2 = np.array([100, 100])
people_3 = np.array([4, 7])
people = (people_1 , people_2 , people_3)
outputs = knn.predict(people)

for output in outputs:
    if  output == 0:
        print("Female♀️")
    else:
        print("male♂️")
        
        
Y_pred = knn.predict(X_test)

confusion_matrix = np.zeros((2,2))

for i in range(180):
    if Y_test[i] == 0 and Y_pred[i] == 0:
        confusion_matrix[0][0] += 1
    elif Y_test[i] == 0 and Y_pred[i] == 1:
        confusion_matrix[0][1] += 1
    elif Y_test[i] == 1 and Y_pred[i] == 0:
        confusion_matrix[1][0] += 1
    elif Y_test[i] == 1 and Y_pred[i] == 1:
        confusion_matrix[1][1] += 1
        
        
confusion_matrix

plt.imshow(confusion_matrix)
plt.colorbar()
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
knn.predict(people)

knn.score(X_test,Y_test)
