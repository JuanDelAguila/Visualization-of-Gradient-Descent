# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams.update({'font.size': 8})

# Defining the domain
x_min, x_max = -150, 150
y_min, y_max = -150, 150

# Defining the function
def f(x,y):
    return (x/5)**2 + y**2


# Defining the derivative of the function
def dfdx(x,y):
    return (2/25)*x + 0*y

def dfdy(x,y):
    return 2*y + 0*x


# Generating 1000 evenly spaced x,y intervals to evaluate the function
x=np.linspace(x_min,x_max,1000)
y=np.linspace(y_min,y_max,1000)


# Starting point
start_x = -100 
start_y = 100 

# Creating a gridspace for the graph and evaluating z
X,Y = np.meshgrid(x,y)
Z=f(X,Y)

# Generating the steps towards minima
number_steps = 100
previous_guess_x = start_x
previous_guess_y = start_y
historic_steps = np.array([start_x,start_y])
step_size = 0.8
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
    historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
    i+=1

# Track the gradient descent
plt.plot(historic_steps[:,0], historic_steps[:,1], color = 'orange', label = "Minimization Steps")

plt.plot(start_x,start_y, 'ko', label = 'Starting Point')
plt.plot(0,0, 'ro', label = 'Minimum')

plt.contourf(X,Y,Z, 500,cmap = 'viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title(r"Minimizing $f(x,y) = \left(\frac{x}{5}\right)^2+y^2$ using Steepest Descent" + "\nwith 100 steps and " + f"step-size {step_size}", size = 10, pad = 10)

plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)

plt.legend()

plt.savefig(f"results/minimize_f_steepest_descent_stepsize_{step_size}.png", dpi = 300)

