import pandas as pd  # data handeling
import numpy as np   # numeriacal computing
import matplotlib.pyplot as plt  # plotting core
import seaborn as sns  # higher level plotting tools

sns.set()

path = 'c:\\python36\\data\\ex1data5.txt'
df_98039 = pd.read_csv(path, header=None, names=['Sqrfeet', 'Price'])
x = df_98039['Sqrfeet']
y = df_98039['Price']
m = len(x)

def h(x,a0,a1) :  # The model function h
    h = a0 + a1*x
    return h

def J(a0, a1, x, y, m):  # The Cost function J
    J = 0
    for i in range(m):
        J += ( h(x[i],a0,a1) - y[i] )**2
    return J/(2.0*m)

def gradJ(a0,a1,x,y,m) :  # The Gradient vector of J, gradJ
    DJa0 = 0; DJa1 = 0;
    for i in range(m):
        DJa0 += (h(x[i],a0,a1) - y[i])
        DJa1 += (h(x[i],a0,a1) - y[i])*x[i]
    gradJ = [DJa0/m, DJa1/m]
    return gradJ

def ap1(a0, a1, gJ, alpha) :  # The update to the parameter vector a, ap1 (a plus 1)
                              # gJ is the gradient vector and alpha is the step length
    a0p1 = a0 - alpha * gJ[0]
    a1p1 = a1 - alpha * gJ[1]
    ap1 = [a0p1, a1p1]
    return ap1

plt.plot(x,y, 'o', color='g', label='raw training data'); plt.legend(fontsize=20);
plt.show()

plt.plot(x/1000,y/1000000, 'o', color='g', label='Simple scaled training data'); plt.legend(fontsize=20);
plt.show()

fig, ax = plt.subplots()
# setup the contour axis
p0 = np.linspace(-8,8, 500)
p1 = np.linspace(-3,3, 500)
pa0, pa1 = np.meshgrid(p0, p1)

# plot the Cost function J
ax.contour(pa0,pa1,J(pa0,pa1,x/1000,y/1000000,m=len(x)) , colors='C0', levels=[i for i in np.arange(0,80,5)])

# starting point
a0 = 6; a1 = 2.2
ax.plot(a0,a1, 'o')

# do the gradient descent loop and plot the progress
for i in range(300):
    a0old, a1old = a0, a1
    a0,a1 = ap1(a0,a1, gJ=gradJ(a0,a1,x/1000,y/1000000,m), alpha=0.1 )
    #print(a0,a1) # the updated parameters
    ax.plot(a0,a1,'o')
    ax.plot([a0old,a0],[a1old,a1], '-') # connect the dots
plt.show()
# the values of the parameters after the optimization run    
print(a0,a1)

fig, ax = plt.subplots()
ax.plot(x/1000,y/1000000, 'o', color='g', label='training data')
ax.plot(x/1000, h(x/1000,a0,a1), label='h(x)')
for i in range(len(x)):
    ax.plot([x[i]/1000, x[i]/1000], [a0 + a1*x[i]/1000,y[i]/1000000], '-', color='c')
plt.legend();
plt.show()

       
print(h(4.501,a0,a1)*1000)
