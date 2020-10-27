import math 
import matplotlib.pyplot as plt
import numpy as np
from constants import heating, rotations

def delta(t,x,v):
    sigma=0.001
    return 1/np.sqrt(2*np.pi*sigma)*np.e**(-(x+v*t)**2/(2*sigma**2))

def f(t,x,v,delta_t,number_t):
    wo=np.pi/(delta_t*number_t)*rotations
    f0 = heating/(number_t*delta_t)
    tau=number_t*delta_t*0.4
    return f0*np.exp(-t**2/tau**2)*delta(t,x,v)*np.cos(wo*t)**2
    
class moving:
    def f0(number_t,number_z,delta_t,delta_z):
        v=(number_z*delta_z)/(number_t*delta_t)
        grid_z = np.array(delta_z*np.arange(number_z))
        grid_t = np.array(delta_t*np.arange(number_t))
        grid = np.zeros((number_t,number_z))
        for i in np.arange(number_t):
            grid[i]=f(grid_t[i]-0.5*(number_t-1)*delta_t,grid_z-0.5*(number_z-1)*delta_z,v,delta_t,number_t)
        return grid
class fix:
    def f0(number_t,number_z,delta_t,delta_z):
        v=0
        grid_z = np.array(delta_z*np.arange(number_z))
        grid_t = np.array(delta_t*np.arange(number_t))
        grid = np.zeros((number_t,number_z))
        for i in np.arange(number_t):
            grid[i]=f(grid_t[i]-0.5*(number_t-1)*delta_t,grid_z-0.5*(number_z-1)*delta_z,0,delta_t,number_t)
        return grid

class no:
    def f0(number_t,number_z,delta_t,delta_z):
        return np.zeros((number_t,number_z))

if __name__ == "__main__":

    a = 23*10**-6
    number_t = 100
    max_t = 10       #[s]
    max_z = 0.2         # [m]     
    delta_t = max_t/number_t
    delta_z = np.sqrt(2*delta_t*a) #stability 
    number_z = int(math.floor(max_z/delta_z))
    plt.imshow(moving.f0(number_t,number_z,delta_t,delta_z),cmap="jet",aspect='auto')   
    plt.colorbar().set_label('temperature [K]')
    #print(moving.f0(number_t,number_z,delta_t,delta_z))