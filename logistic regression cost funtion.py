import numpy as np   # numeriacal computing
import matplotlib.pyplot as plt  # plotting core
import seaborn as sns  # higher level plotting tools

sns.set()

def g(z) :  # sigmoid function
    return 1/(1 + np.exp(-z))

z = np.linspace(-10,10,100)
fig, ax = plt.subplots()
ax.plot(z, g(z)) 
ax.set_title('Sigmoid Function 1/(1 + exp(-z))', fontsize=24)
ax.annotate('Convex', (-7.5,0.2), fontsize=18 )
ax.annotate('Concave', (3,0.8), fontsize=18 )
plt.show()

z = np.linspace(-10,10,100)
plt.plot(z, -np.log(g(z)))
plt.title("Log Sigmoid Function -log(1/(1 + exp(-z)))", fontsize=24)
plt.annotate('Convex', (-2.5,3), fontsize=18 )
plt.show()

x = np.linspace(-10,10,50)
plt.plot(g(x), -np.log(g(x)))
plt.title("h(x) vs J(a)=-log(h(x)) for y = 1", fontsize=24)
plt.xlabel('h(x)')
plt.ylabel('J(a)')
plt.show()

x = np.linspace(-10,10,50)
plt.plot(g(x), -np.log(1-g(x)))
plt.title("h(x) vs J(a)=-log(1-h(x)) for y = 0", fontsize=24)
plt.xlabel('h(x)')
plt.ylabel('J(a)')
plt.show()
