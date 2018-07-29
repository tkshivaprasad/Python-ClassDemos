import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True

path = 'c:\\python36\\data\\ex1data5.txt'
data = pd.read_csv(path, header=None, names=['Sqrfeet', 'Price'])
data.head()


# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[0:95,0:cols-1]  
y = data.iloc[0:95,cols-1:cols]
#print(X)
#print(y)

# convert from data frames to numpy matrices
X = np.matrix(X.values)  
y = np.matrix(y.values)  
theta = np.matrix(np.array([0,0]))
#print(X)


from sklearn import linear_model  
model = linear_model.LinearRegression()  
model.fit(X, y)

x_predict=np.matrix([1, 900]) 

predict = model.predict(x_predict)
print("for 900 sq feet, the predictred price is ", predict)

x_predict=np.matrix([1, 7000])
predict = model.predict(x_predict)
print("for 7000 sq feet, the predictred price is ", predict)
print(model.score(X,y))

x = np.array(X[:, 1].A1)  
f = model.predict(X).flatten()


fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Sqrfeet, data.Price, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Sqrfeet')  
ax.set_ylabel('Price')  
ax.set_title('Predicted Price vs. Square feet')
plt.show()


