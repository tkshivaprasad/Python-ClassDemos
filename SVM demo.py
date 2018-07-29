import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
from sklearn import svm
svc = svm.SVC(kernel='rbf', C=100,gamma="auto").fit(X, y)
#change kernel-linear,rbf,poly,sigmoid,
#change value of gamma to 0, 10, 100 - higher the value of gamma - overfit the data
#change the value of C to 10, 100
#plotting the decision boundary
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.01
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole gid
fig, ax = plt.subplots()
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
ax.set_title('Support Vector Machine')
plt.show()
