from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(1, 11).reshape(10, 1)
y = np.array([7, 8, 7, 13, 16, 15, 19, 23, 18, 21]).reshape(10, 1)

plt.plot(X, y, 'ro')
plt.show()
# simple linear regression
model = LinearRegression()

model.fit(X, y)

a = model.coef_ * X + model.intercept_

plt.plot(X, y, 'ro', X, a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()

print(model.score(X, y))

#hypothesis - quadratic
X = np.c_[X, X**2]
model.fit(X, y)
x = np.arange(1, 11, 0.1)
x = np.c_[X, X**2]
a = np.dot(X, model.coef_.transpose()) + model.intercept_

plt.plot(X[:, 0], y, 'ro', X[:, 0], a)
plt.show()
print(model.score(X, y))

#hypothesis - polynomial features
X = np.arange(1, 11)
X = np.c_[X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9]
x = np.arange(1, 11, 0.1)
x = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]

model.fit(X, y)
a = np.dot(x, model.coef_.transpose()) + model.intercept_

plt.plot(X[:, 0], y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()
print(model.score(X, y))

#with test data set
X = np.arange(1, 16)
y = np.append(y, [24, 23, 22, 26, 22])

plt.plot(X, y, 'ro')
plt.show()
plt.plot(X, y, 'ro', x[:, 0], a)
axes = plt.gca()
axes.set_ylim([0, 30])
plt.show()

