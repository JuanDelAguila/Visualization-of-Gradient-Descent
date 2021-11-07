# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams.update({'font.size': 8})

# Defining the domain
x_min, x_max = -2.5, 2.5
y_min, y_max = -2.5, 2.5


# Defining the function
def f(x,y):
    return (-x**2-y**2)*np.exp(-x**2-y**2)+(-x**5-y**5)*np.exp(-x**2-y**2)

# Defining the derivative of the function
def dfdx(x,y):
    return x*np.exp(-x**2-y**2)*(2*x**5-5*x**3+2*x**2-2+2*y**5+2*y**2)

def dfdy(x,y):
    return y*np.exp(-x**2-y**2)*(2*y**5-5*y**3+2*y**2-2+2*x**5+2*x**2)


# Generating 1000 evenly spaced x,y intervals to evaluate the function
x=np.linspace(x_min,x_max,1000)
y=np.linspace(y_min,y_max,1000)

# Creating a gridspace for the graph and evaluating z
X,Y = np.meshgrid(x,y)
Z=f(X,Y)

# Momentum algorithm
def momentum(start_x, start_y):
    number_steps = 100
    step_size = 3
    B = 0.9
    previous_guess_x = start_x
    previous_guess_y = start_y
    previous_vdx = 0
    previous_vdy = 0
    historic_steps = np.array([start_x,start_y])
    i = 1
    while i <= number_steps:
        new_vdx = B*previous_vdx + (1-B)*dfdx(previous_guess_x, previous_guess_y)
        new_vdy = B*previous_vdy + (1-B)*dfdy(previous_guess_x, previous_guess_y)
        new_guess_x = previous_guess_x - step_size*new_vdx
        new_guess_y = previous_guess_y - step_size*new_vdy
        previous_guess_x = new_guess_x
        previous_guess_y = new_guess_y
        historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
        i+=1
    return historic_steps

# Plotting minimization for different starting points
starting_points = [[2,1], [0.3,0.5], [0.5,0.5], [-0.1, -0.3]]
colors = ["magenta", "orange", "red", "black"]
for (start_x, start_y), color in zip(starting_points, colors):
    historic_steps = momentum(start_x, start_y)
    plt.plot(historic_steps[:,0], historic_steps[:,1], color = color, label = f"Starting Point: [{start_x},{start_y}]")
    plt.plot(start_x,start_y, color = color, marker = 'o')

# Plotting the rest of the graph
plt.contourf(X,Y,Z, 500,cmap = 'viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
g = r"$g(x,y) = -(x^2+y^2)e^{-x^2-y^2}-(x^5+y^5)e^{-x^2-y^2}$"
plt.title(r"Minimizing "+ g + "\n using the Momentum Method with 100 steps, step-size 7,\nand momentum parameter 0.9", size = 10, pad = 5)

plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)

plt.legend(loc = "lower right")
plt.margins(x=50)

plt.savefig(f"results/minimize g with momentum.png", dpi = 300)

