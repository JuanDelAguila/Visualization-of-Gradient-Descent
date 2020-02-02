#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import random
#####################

#Defining the function
def f(x,y):
    return (x/5)**2 + y**2
#####################


#Defining the derivative of the function
def dfdx(x,y):
    return (2/25)*x + 0*y

def dfdy(x,y):
    return 2*y + 0*x
#####################


#Generating 1000 evenly spaced x,y intervals to evaluate the function
x=np.linspace(-200,200,1000)
y=np.linspace(-200,200,1000)
#####################


#generating a random innitial guess
randx = -100 #random.choice(x)
randy = 100#random.choice(y)
#####################


#creating a gridspace for the graph and evaluating z
X,Y = np.meshgrid(x,y)

Z=f(X,Y)
#####################


#generating the steps towards minima
number_steps = 101
step_size = 7
B = 0.9
previous_guess_x = randx
previous_guess_y = randy
previous_vdx = 0
previous_vdy = 0
historic_steps = np.array([randx,randy])
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
print (historic_steps)
######################


contour_plot = plt.figure(2)

#track the gradient descent by points

a = 0
a = int(a)
print(len(historic_steps))
for a in range(0,len(historic_steps)-1):
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]], color='chartreuse')
    a+=1
a,b = historic_steps.T
plt.scatter(a,b, c='black', s=10)

plt.plot(randx,randy,marker='o', color='black')
######################


plt.contourf(X,Y,Z, 500,cmap = 'coolwarm')
plt.colorbar()
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.xlim(-200,200)
plt.ylim(-200,200)

plt.show()
