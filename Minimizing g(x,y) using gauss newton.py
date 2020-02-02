#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import random
#####################

#Defining the function
def f(x,y):
    return (-x**2-y**2)*np.exp(-x**2-y**2)+(-x**5-y**5)*np.exp(-x**2-y**2)
#####################


#Defining the derivatives of the function
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
#####################


#Generating 1000 evenly spaced x,y intervals to evaluate the function
x=np.linspace(-3,3,1000)
y=np.linspace(-3,3,1000)
#####################


#generating a random innitial guess
randx = -2#random.choice(x)
randy = 2#random.choice(y)
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
step_size = 1
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    hess = np.array([[df2dx2(previous_guess_x,previous_guess_y),df2dxy(previous_guess_x,previous_guess_y)],
                    [df2dyx(previous_guess_x,previous_guess_y),df2dy2(previous_guess_x,previous_guess_y)]])
    
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*np.linalg.inv(hess)@grad 
    if i == 1:
        print(new_guess)
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
    historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
    i+=1
print(historic_steps)
#print(np.linalg.inv(hess))
#print (historic_steps)
######################


contour_plot = plt.figure(2)


#track the gradient descent by points

a = 0
a = int(a)
print(len(historic_steps))

for a in range(0,len(historic_steps)-1):
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]],color='chartreuse')
    a+=1
    
#a,b = historic_steps.T
#plt.scatter(a,b, c='black', s=10)

plt.plot(randx,randy,marker='o', color='chartreuse')
######################


plt.contourf(X,Y,Z, 500,cmap = 'coolwarm')
plt.colorbar()
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.xlim(-3,3)
plt.ylim(-3,3)

#Tracking the progress of different starting points

#########################################################################################################
#########################################################################################################
#########################################################################################################

#Start at: (-0.2,1.2)
randx = -0.2#random.choice(x)
randy = 1.2#random.choice(y)
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
step_size = 1
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    hess = np.array([[df2dx2(previous_guess_x,previous_guess_y),df2dxy(previous_guess_x,previous_guess_y)],
                    [df2dyx(previous_guess_x,previous_guess_y),df2dy2(previous_guess_x,previous_guess_y)]])
    
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad @np.linalg.inv(hess)
    if i == 1:
        print(new_guess)
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
    historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
    i+=1
print(historic_steps)
#print(np.linalg.inv(hess))
#print (historic_steps)
######################


contour_plot = plt.figure(2)


#track the gradient descent by points

a = 0
a = int(a)
print(len(historic_steps))

for a in range(0,len(historic_steps)-1):
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]],color='magenta')
    a+=1
    

plt.plot(randx,randy,marker='o', color='magenta')
######################
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Start at: (0.1,0.1)
randx = 0.1#random.choice(x)
randy = 0.1#random.choice(y)
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
step_size = 1
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    hess = np.array([[df2dx2(previous_guess_x,previous_guess_y),df2dxy(previous_guess_x,previous_guess_y)],
                    [df2dyx(previous_guess_x,previous_guess_y),df2dy2(previous_guess_x,previous_guess_y)]])
    
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad @np.linalg.inv(hess)
    if i == 1:
        print(new_guess)
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
    historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
    i+=1
print(historic_steps)
#print(np.linalg.inv(hess))
#print (historic_steps)
######################


contour_plot = plt.figure(2)


#track the gradient descent by points

a = 0
a = int(a)
print(len(historic_steps))

for a in range(0,len(historic_steps)-1):
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]],color='deepskyblue')
    a+=1
    

plt.plot(randx,randy,marker='o', color='deepskyblue')
######################
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Start at: (-0.1,-0.1)
randx = 0#random.choice(x)
randy = -0.2#random.choice(y)
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
step_size = 1
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    hess = np.array([[df2dx2(previous_guess_x,previous_guess_y),df2dxy(previous_guess_x,previous_guess_y)],
                    [df2dyx(previous_guess_x,previous_guess_y),df2dy2(previous_guess_x,previous_guess_y)]])
    
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad @np.linalg.inv(hess)
    if i == 1:
        print(new_guess)
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
    historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
    i+=1
print(historic_steps)
#print(np.linalg.inv(hess))
#print (historic_steps)
######################


contour_plot = plt.figure(2)


#track the gradient descent by points

a = 0
a = int(a)
print(len(historic_steps))

for a in range(0,len(historic_steps)-1):
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]],color='darkorchid')
    a+=1
    

plt.plot(randx,randy,marker='o', color='darkorchid')
######################
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Start at: (0.7,0.7)
randx = 0.7#random.choice(x)
randy = 0.7 #random.choice(y)
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
step_size = 1
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    hess = np.array([[df2dx2(previous_guess_x,previous_guess_y),df2dxy(previous_guess_x,previous_guess_y)],
                    [df2dyx(previous_guess_x,previous_guess_y),df2dy2(previous_guess_x,previous_guess_y)]])
    
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad @np.linalg.inv(hess)
    if i == 1:
        print(new_guess)
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
    historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
    i+=1
print(historic_steps)
#print(np.linalg.inv(hess))
#print (historic_steps)
######################


contour_plot = plt.figure(2)


#track the gradient descent by points

a = 0
a = int(a)
print(len(historic_steps))

for a in range(0,len(historic_steps)-1):
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]],color='gold')
    a+=1
    

plt.plot(randx,randy,marker='o', color='gold')
######################
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Start at: (0,0)
randx = 0#random.choice(x)
randy = 0#random.choice(y)
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
step_size = 1
i = 1
while i <= number_steps:
    grad = np.array([dfdx(previous_guess_x,previous_guess_y),dfdy(previous_guess_x,previous_guess_y)])
    hess = np.array([[df2dx2(previous_guess_x,previous_guess_y),df2dxy(previous_guess_x,previous_guess_y)],
                    [df2dyx(previous_guess_x,previous_guess_y),df2dy2(previous_guess_x,previous_guess_y)]])
    
    new_guess = np.array([previous_guess_x, previous_guess_y]) - step_size*grad @np.linalg.inv(hess)
    if i == 1:
        print(new_guess)
    previous_guess_x = new_guess[0]
    previous_guess_y = new_guess[1]
    historic_steps = np.vstack((historic_steps,np.array([previous_guess_x, previous_guess_y])))
    i+=1
print(historic_steps)
#print(np.linalg.inv(hess))
#print (historic_steps)
######################


contour_plot = plt.figure(2)


#track the gradient descent by points

a = 0
a = int(a)
print(len(historic_steps))

for a in range(0,len(historic_steps)-1):
    plt.plot([historic_steps[a,0],historic_steps[a+1,0]],[historic_steps[a,1],historic_steps[a+1,1]],color='black')
    a+=1
    

plt.plot(randx,randy,marker='o', color='black')
######################
#########################################################################################################
#########################################################################################################
#########################################################################################################

plt.show()


