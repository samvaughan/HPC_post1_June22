import math
def python_sine(array):
    results=[math.sin(value) for value in array]
    return results


import numpy as np
def numpy_sine(array):
    results=np.sin(array)
    return results


from numba import jit
@jit
def numba_sine(array):
    results=np.zeros_like(array)
    for i, value in enumerate(array):
        results[i]=math.sin(value)
    return results

def time_function(function, array, string):
    """
    A simple function to time the call of a function. Array should be the only argument of that function, and 
    string is something that is printed before the elapsed time is shown"""
    import time
    t0=time.clock()
    function(array)
    t1=time.clock()

    print '{}: {} seconds'.format(string, t1-t0)


if __name__=='__main__':
    import time
    array=np.random.rand(1000000)

    time_function(python_sine, array, 'Python function')
    time_function(numpy_sine, array, 'Numpy function')

    #Run it once to compile
    numba_sine(array)
    time_function(numba_sine, array, 'Numba function')