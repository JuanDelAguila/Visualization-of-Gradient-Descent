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

# Defining the derivatives of the function
def dfdx(x,y):
    return x*np.exp(-x**2-y**2)*(2*x**5-5*x**3+2*x**2-2+2*y**5+2*y**2)

def dfdy(x,y):
    return y*np.exp(-x**2-y**2)*(2*y**5-5*y**3+2*y**2-2+2*x**5+2*x**2)

def df2dx2(x,y):
    return np.exp(-x**2-y**2 )*(-4*x**7+22*x**5-4*x**4-20*x**3+10*x**2-2-4*(x**2)*(y**5)-4*(x**2)*(y**2)+2*y**2+2*y**5) 

def df2dxy(x,y):
    return y*x*np.exp(-x**2-y**2)*(-4*y**5+10*y**3-4*y**2+8-4*x**2+10*x**3-4*x**5)

def df2dy2(x,y):
    return np.exp(-x**2-y**2 )*(-4*y**7+22*y**5-4*y**4-20*y**3+10*y**2-2-4*(y**2)*(x**5)-4*(y**2)*(x**2)+2*x**2+2*x**5)

def df2dyx(x,y):
    return y*x*np.exp(-x**2-y**2)*(-4*x**5+10*x**3-4*x**2+8-4*y**2+10*y**3-4*y**5)

# Generating 1000 evenly spaced x,y intervals to evaluate the function
x=np.linspace(x_min,x_max,1000)
y=np.linspace(y_min,y_max,1000)

# Creating a gridspace for the graph and evaluating z
X,Y = np.meshgrid(x,y)
Z=f(X,Y)

# Newton algorithm
def newton (start_x, start_y):
    number_steps = 100
    previous_guess_x = start_x
    previous_guess_y = start_y
    historic_steps = np.array([start_x,start_y])
    step_size = 0.1
    i = 1
    while i <= number_steps:
        grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
        hess = np.array([[df2dx2(previous_guess_x,previous_guess_y),df2dxy(previous_guess_x,previous_guess_y)],
                        [df2dyx(previous_guess_x,previous_guess_y),df2dy2(previous_guess_x,previous_guess_y)]])
        
        new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*np.linalg.inv(hess)@grad 
        previous_guess_x = new_guess[0]
        previous_guess_y = new_guess[1]
        historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
        i+=1
    return historic_steps

contour_plot = plt.figure(2)

# Plotting minimization for different starting points
starting_points = [[2,1], [0.3,0.5], [0.5,0.5], [-0.1, -0.3], [0.4,1.8]]
colors = ["magenta", "orange", "black", "red", "cyan"]
for (start_x, start_y), color in zip(starting_points, colors):
    historic_steps = newton (start_x, start_y)
    plt.plot(historic_steps[:,0], historic_steps[:,1], color = color, label = f"Starting Point: [{start_x},{start_y}]")
    plt.plot(start_x,start_y, color = color, marker = 'o')

# Plotting the rest of the graph
plt.contourf(X,Y,Z, 500,cmap = 'viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
g = r"$g(x,y) = -(x^2+y^2)e^{-x^2-y^2}-(x^5+y^5)e^{-x^2-y^2}$"
plt.title(r"Minimizing "+ g + "\n using Newton's method with 100 steps and step-size 0.1", size = 10, pad = 10)

plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)

plt.legend(loc = "lower right")

plt.savefig(f"results/minimize_g_newton.png", dpi = 300)

