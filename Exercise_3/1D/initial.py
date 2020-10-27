import numpy as np
from constants import u0, std



def initial_analyticaly(x,delta_z,number_z):
    return u0*np.exp(-x**2/std**2)

class normal:
    def T0(number_z,delta_z):
        grid = np.array(delta_z*np.arange(number_z))
        grid = grid-0.5*(number_z-1)*delta_z
        return initial_analyticaly(grid,delta_z,number_z)

class const:
    def T0(number_z,delta_z):
        grid = np.array(delta_z*np.arange(number_z))
        return grid*0
        