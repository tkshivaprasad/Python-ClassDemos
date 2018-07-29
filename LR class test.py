import pandas as pd  # data handeling
import numpy as np   # numeriacal computing
import matplotlib.pyplot as plt  # plotting core
import seaborn as sns  # higher level plotting tools

sns.set()

path = 'C:\\Users\\TKS\\Desktop\\data\\ex1data4.txt'
df_98039 = pd.read_csv(path, header=None, names=['Sqrfeet', 'Price'])
x = df_98039['Sqrfeet']
y = df_98039['Price']

def J(a0, a1, x, y, m):
    J = 0
    for i in range(m):
        J += ((a0 + a1*x[i]) - y[i] )**2
    return J

fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')
ax.plot(x, 5000 + 560*x, label='h(x) = 5000 + 560*x')
for i in range(len(x)):
    ax.plot([x[i], x[i]], [5000 + 560*x[i],y[i]], '-', color='c')
plt.legend();
plt.show()

fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')
a1 = np.linspace(100,500,8)
for i in range(len(a1)):
    ax.plot(x, 150 + a1[i]*x, label='a1 = %.2f' %a1[i] )
plt.legend();
plt.show()

fig, ax = plt.subplots()
a = np.linspace(100,500,8)
ax.plot(a, J(150,a,x,y,m=len(x)), c='C0')
for i in range(len(a1)):
    ax.plot(a1[i], J(150,a1[i],x,y,m=len(x)), 'o', label='J(a0,%.2f)' %a1[i])
plt.legend();
plt.show()

x1 = np.linspace(0,5,5)

y1 = [ 0.21378624, 1.97217916, 2.36737375, 5.13718724, 6.26470731]
a0 = np.linspace(-2,2, 100)
a1 = np.linspace(0,2.2, 100)
aa0, aa1 = np.meshgrid(a0, a1)
plt.contour(aa0,aa1,J(aa0,aa1,x1,y1,m=len(x1)) , colors='C0', levels=[i for i in np.arange(0,80,3.75)])
plt.show()
