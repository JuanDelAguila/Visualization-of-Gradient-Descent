#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import random
#####################

#Defining the function
def f(x,y):
    return (-x**2-y**2)*np.exp(-x**2-y**2)+(-x**5-y**5)*np.exp(-x**2-y**2)
#####################


#Defining the derivative of the function
def dfdx(x,y):
    return x*np.exp(-x**2-y**2)*(2*x**5-5*x**3+2*x**2-2+2*y**5+2*y**2)

def dfdy(x,y):
    return y*np.exp(-x**2-y**2)*(2*y**5-5*y**3+2*y**2-2+2*x**5+2*x**2)
#####################


#Generating 1000 evenly spaced x,y intervals to evaluate the function
x=np.linspace(-3,3,1000)
y=np.linspace(-3,3,1000)
#####################


#generating a random innitial guess
randx = -2 #random.choice(x)
randy = 2 #random.choice(y)
#####################


#creating a gridspace for the graph and evaluating z
X,Y = np.meshgrid(x,y)

Z=f(X,Y)
#####################


#generating the steps towards minima
number_steps = 100
previous_guess_x = randx
previous_guess_y = randy
historic_steps = np.array([randx,randy])
step_size = 0.2
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
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

plt.plot(randx,randy,marker='o', color='chartreuse')

######################
#Different Starting points
######################


#Innitial guess (0,0)
randx = 0
randy = 0
#####################

#generating the steps towards minima
number_steps = 100
previous_guess_x = randx
previous_guess_y = randy
historic_steps = np.array([randx,randy])
step_size = 0.2
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
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
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]], color='deepskyblue')
    a+=1
a,b = historic_steps.T
plt.scatter(a,b, c='black', s=10)

plt.plot(randx,randy,marker='o', color='deepskyblue')



#Innitial guess (0.1,0.1)
randx = 0.1
randy = 0.1
#####################

#generating the steps towards minima
number_steps = 100
previous_guess_x = randx
previous_guess_y = randy
historic_steps = np.array([randx,randy])
step_size = 0.2
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
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
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]], color='magenta')
    a+=1
a,b = historic_steps.T
plt.scatter(a,b, c='black', s=10)

plt.plot(randx,randy,marker='o', color='magenta')


#Innitial guess (-0.1,-0.1)
randx = -0.1
randy = -0.1
#####################

#generating the steps towards minima
number_steps = 100
previous_guess_x = randx
previous_guess_y = randy
historic_steps = np.array([randx,randy])
step_size = 0.2
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
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
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]], color='darkgreen')
    a+=1
a,b = historic_steps.T
plt.scatter(a,b, c='black', s=10)

plt.plot(randx,randy,marker='o', color='darkgreen')

plt.contourf(X,Y,Z, 500,cmap = 'coolwarm')
plt.colorbar()
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.xlim(-3,3)
plt.ylim(-3,3)

#contour_plot.show()
plt.show()
