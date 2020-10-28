import math 
import matplotlib.pyplot as plt
import numpy as np


def delta(t,x,v):
    sigma=0.001
    return 1/np.sqrt(2*np.pi*sigma)*np.e**(-(x+v*t)**2/(2*sigma**2))

def f(t,x,v,delta_t,number_t,wo,f0,tau):
    return f0*np.exp(-t**2/tau**2)*delta(t,x,v)*np.cos(wo*t)**2
    
def moving(number_t,number_z,delta_t,delta_z,wo,f0,tau):
    v=(number_z*delta_z)/(number_t*delta_t)
    grid_z = np.array(delta_z*np.arange(number_z))
    grid_t = np.array(delta_t*np.arange(number_t))
    grid = np.zeros((number_t,number_z))
    for i in np.arange(number_t):
        grid[i]=f(grid_t[i]-0.5*(number_t-1)*delta_t,grid_z-0.5*(number_z-1)*delta_z,v,delta_t,number_t,wo,f0,tau)
    return grid

def fix(number_t,number_z,delta_t,delta_z,wo,f0,tau):
    v=0
    grid_z = np.array(delta_z*np.arange(number_z))
    grid_t = np.array(delta_t*np.arange(number_t))
    grid = np.zeros((number_t,number_z))
    for i in np.arange(number_t):
        grid[i]=f(grid_t[i]-0.5*(number_t-1)*delta_t,grid_z-0.5*(number_z-1)*delta_z,0,delta_t,number_t,wo,f0,tau)
    return grid

def no_source(number_t,number_z,delta_t,delta_z):
        return np.zeros((number_t,number_z))

