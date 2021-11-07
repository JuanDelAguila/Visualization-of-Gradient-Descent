# Visualization-of-Gradient-Descent

This is a personal investigation into the efficiency and accuracy of different gradient descent algorithms. Three algorithms were tested: Steepest Descent, Newton's Method, and the Momentum Method. The python scripts generate visualizations of various approximation attempts made with these algorithms from various starting points. The two functions tested are:
- f(x,y) = (x/5)^2 + y^2
- g(x,y) = (-x^2-y^2)*e^(-x^2-y^2) + (-x^5-y^5)*e^(-x^2-y^2) 
 
## Results
![](results/minimize_f_steepest_descent_stepsize_0.4.png)
![](results/minimize_f_steepest_descent_stepsize_0.8.png)
![](results/minimize_f_newton.png)
![](results/minimize_f_momentum.png)
![](results/minimize_g_steepest_descent.png)
![](results/minimize_g_newton.png)
![](results/minimize_g_momentum.png)