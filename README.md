# py_func_fitting
This is a one-dimensional data fitting algorithm, which can be used when you want to fit a curve with a function but the function expression is unknown. The algorithm uses optimization algorithm to optimize expression binary tree. This program is written in python3.

Before using this program, make sure you have installed SciPy v1.1.0.

And there is something you need to change in from scipy.optimize.curve_fit. Because I add a new return function of it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This file in path:your python path\lib\site-packages\scipy\optimize\minpack.py. 
Or, with some python IDE, you can just use its function called 'go to defination'.  

At line 513, I add a new variable named my_return, and = False.
```
,my_return = False,
```
![1](https://github.com/shashadehuajiang/py_func_fitting/blob/master/pics/1.PNG)


From line 796 to 797, if my_return: return popt,cost
```
If my_return:
    return popt,cost
```
![2](https://github.com/shashadehuajiang/py_func_fitting/blob/master/pics/2.PNG)



That's all what you need to change.
Now, let's start our journey.


At the end of mainv1.py, there is a class called Evolution_f(xdata,ydata,function_number=100,episode = 100,alpha = 0.1,hintfunction='').
xdata and ydata are the main input (the data in), and the function_number and episode are the paras of optimization algorithm, like genetic algorithm. alpha controls the loss function: loss = alpha*function_length + (1-alpha)*msecost. hintfunction is a function that you think can be used.

The test() function at the end of mainv1.py is an example to show you how to use the program.
```
def test():
    x = np.linspace(0, 1, 100)
    y = 10*np.log(x+0.1)+2.3

    plt.plot(x, y, '.')
    plt.show()
    
    #xdata,ydata,function_number=100,episode = 100,alpha = 0.1,hintfunction=''
    ev1 = Evolution_f(x,y_data,2000,30,0.1,'')
    outfunc = ev1.start()

    return outfunc
```
First, we generate some dada using function 10*np.log(x+0.1)+2.3. Then we give the xdata and ydtata to Evolution_f. Evolution_f would finally return an optimized function expression.
Now let's start the program.
```
python ./mainv1.py
```
Here is the final return:
```
(((log((b+x)))+c)*a)
```
It's kind of cool,right?
