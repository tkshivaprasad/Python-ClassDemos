import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True

path = 'c:\\python36\\data\\ex1data3.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms','Price'])
data2.head()

data2 = (data2 - data2.mean()) / data2.std()  #feature normalization
data2.head()
data2.describe()

def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]  
X2 = data2.iloc[:,0:cols-1]  
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)  
y2 = np.matrix(y2.values)  
theta2 = np.matrix(np.array([0,0,0])) 

val = computeCost(X2, y2, theta2)
print (val)

def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

# initialize variables for learning rate and iterations
alpha = 0.01  
iters = 1000

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters) 

val = computeCost(X2, y2, g2)
print(val)


fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), cost2, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')
plt.show()

size1 = (2300 - 2000.68) / 794.70
bd = (3 - 3.17) / 0.76
predict = g2[0,0]+size1*g2[0,1]+bd*g2[0,2]
predicted_price=predict*125039.9 + 340412.66
print (predicted_price)

