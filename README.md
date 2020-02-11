# py_func_fitting
This is a one-dimensional data fitting algorithm, which can be used when you want to fit a curve with a function but the function expression is unknown. This program is written in python3.

Before using this program, make sure you have installed SciPy v1.1.0.

And there is something you need to change in from scipy.optimize.curve_fit. Because I add a new return function of it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This file in path:your python path\lib\site-packages\scipy\optimize\minpack.py. 
Or, with some python IDE, you can just use its function called 'go to defination'.  

At line 513, I add a new variable named my_return, and = False.

![1](https://github.com/shashadehuajiang/py_func_fitting/blob/master/pics/1.PNG)


From line 796 to 797, if my_return: return popt,cost

![2](https://github.com/shashadehuajiang/py_func_fitting/blob/master/pics/2.PNG)



That's all what you need to change.
Now, let's start our journey.


