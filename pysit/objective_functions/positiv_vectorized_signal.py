import numpy as np

__all__ = ['PositivVectorizedSignal','get_function']

__docformat__ = "restructuredtext en"

ep = 2.2204e-16

def identity(x):
    return np.asarray(x, dtype=float)

def identity_gradient(x):
    return np.ones_like(np.copy(x), dtype=float)

def smoothMax(x):
    x_pos = 0.5 * (x + np.sqrt(x**2 + (ep * np.max(x))**2))
    x_neg = 0.5 * (-x + np.sqrt(x**2 + (ep * np.max(-x))**2))
    return x_pos, x_neg

def smoothMax_gradient(x):
    dx_pos = -0.5 * (1 + x/(np.sqrt(x**2 + (ep * np.max(x))**2))) #* (np.exp(d/np.max(d))-0.8) #(d**2)
    dx_neg = 0.5 * (1 - x/(np.sqrt(x**2 + (ep * np.max(-x))**2))) #* (np.exp(d/np.max(d))-0.8) #(d**2)
    return dx_pos, dx_neg

def get_function(func_id):
    """
    Return function from its id

    Parameter
    ---------
    func_id : str
        function id

    Return
    ------
    Callable object
    """
    switcher = {
        'id': identity,
        'smooth_max': smoothMax
    }

    switcher_gradient = {
        'id': identity_gradient,
        'smooth_max': smoothMax_gradient
    }

    return switcher.get(func_id, None), switcher_gradient.get(func_id, None)


 