import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True

path = 'c:\\python36\\data\\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
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

x_predict=np.matrix([1, 4.5]) #data.iloc[96,0:cols-1]
#print (x_predict)
#x_predict=np.matrix(x_predict.values)
predict = model.predict(x_predict)
print(predict*10000)

x = np.array(X[:, 1].A1)  
f = model.predict(X).flatten()

# The coefficients
print('Coefficients: \n', model.coef_)


fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


