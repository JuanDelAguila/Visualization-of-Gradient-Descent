import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams.update({'font.size': 8})

# Define the function
def f(x,y):
    return (x/5)**2 + y**2

# Evaluate the function in the chosen domain
x = np.linspace(-150,150,1000)
y = np.linspace(-150,150,1000)
X,Y = np.meshgrid(x,y)
Z=f(X,Y)

# Create a 3D plot of the function
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_surface(X, Y, Z, cmap = cm.viridis, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title(r"$f(x,y) = \left(\frac{x}{5}\right)^2+y^2$", size = 10, pad = 10)
plt.savefig(f"results/f_x_y.png", dpi = 300)