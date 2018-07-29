import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True

path = 'c:\\python36\\data\\ex1data3.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms','Price'])
data2.head()

data2 = (data2 - data2.mean()) / data2.std()
# append a ones column to the front of the data set
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]  
X2 = data2.iloc[:,0:cols-1]  
y2 = data2.iloc[:,cols-1:cols]
#print(X)
#print(y)

# convert from data frames to numpy matrices
X2 = np.matrix(X2.values)  
y2 = np.matrix(y2.values)  
theta = np.matrix(np.array([0,0,0]))
#print(X)


from sklearn import linear_model  
model = linear_model.LinearRegression()  
model.fit(X2, y2)

x_predict=np.matrix([1, 900, 2]) 

predict = model.predict(x_predict)
print("for 900 sq feet and 2 bedrooms, the predictred price is ", predict)

x_predict=np.matrix([1, 7000, 4])
predict = model.predict(x_predict)
print("for 7000 sq feet and 4 bedrooms, the predictred price is ", predict)


print(model.score(X2,y2))
