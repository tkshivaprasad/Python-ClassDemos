import pandas as pd  # data handling
import numpy as np   # numerical computing
import matplotlib.pyplot as plt  # plotting core
import seaborn as sns  # higher level plotting tools

sns.set()

# some simple random (close to linear) data
x = np.linspace(0,5,5)
#y = np.linspace(0,5,5) + 1 + np.random.randn(5)
y = [ 0.21378624, 1.97217916, 2.36737375, 5.13718724, 6.26470731]

fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')
ax.plot(x, .1 + 1.2*x, label='h(x) = .1 + 1.2*x')
for i in range(len(x)):
    ax.plot([x[i], x[i]], [.1 + 1.2*x[i],y[i]], '-', color='c')
plt.legend();
plt.show()
