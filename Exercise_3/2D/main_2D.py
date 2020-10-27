import numpy as np
import math 
import matplotlib.pyplot as plt
from boundary_2d import dirichlet, periodic_horizontal
from source_2D import source
from constants_2D import u0, a, r
number_t = 5
max_t = 5       #[s]
max_z = 0.20         # [m]    



def run(number_t, max_t, max_z):        
    #define variables to make the code clearer
    delta_t = max_t/number_t
    delta_z = np.sqrt(delta_t*a/r) #stability 
    number_z = int(math.floor(max_z/delta_z))
    
    #import initial condition
    T=np.zeros((number_t,number_z**2))
    T[0]=np.ones(number_z**2)*u0
        
    #imort matrix A and vector C coresponding to boundary
    A=dirichlet.A(number_z) 
    C=dirichlet.C(number_z)       
    
    #import source
    F=source(number_z)
    
    #Solving the linear equation sistem for each time step
    for i in np.arange(number_t-1):
        T[i+1]=np.dot(A,T[i])+C+F
    T_2D=np.zeros((number_t,number_z,number_z))
    for i in np.arange(number_t):
        for j in np.arange(number_z**2):
            T_2D[i][math.floor(j/number_z)][j%number_z]=T[i][j]
    
    
    #plot
    plt.imshow(np.transpose(T_2D[9]),cmap="jet",aspect='auto')   
    z_ticks_stepsize=0.1
    plt.yticks(np.arange((max_z)/z_ticks_stepsize)*number_z/(max_z/z_ticks_stepsize),np.round((np.array(-np.arange((max_z/z_ticks_stepsize)))*z_ticks_stepsize)+max_z*0.5,2))
    plt.xticks(np.arange((max_z)/z_ticks_stepsize)*number_z/(max_z/z_ticks_stepsize),-np.round((np.array(-np.arange((max_z/z_ticks_stepsize)))*z_ticks_stepsize)+max_z*0.5,2))
    plt.xlabel('position [m]')
    plt.ylabel('position [m]')
    plt.colorbar().set_label('temperature [K]')
    plt.show()  
  
number_t = 70
max_t = 700       #[s]
max_z = 0.4         # [m]    



run(number_t, max_t, max_z)