import numpy as np
import math 
import matplotlib.pyplot as plt

from boundary import dirichlet, neumann, cauchy, periodic 
from initial import normal, const
from source import moving, fix, no
from analytical import analytical_dirichlet
from constants import a, u0, const_dirichlet
    
    
def run(number_t, max_t, max_z,boundary_type,initial_type,source_typ):        
    #define variables to make the code clearer
    delta_t = max_t/number_t
    delta_z = np.sqrt(2*delta_t*a) #stability 
    number_z = int(math.floor(max_z/delta_z))
    print(number_z)
    
    #import initial condition
    T=np.zeros((number_t,number_z))
    T[0]=initial_type.T0(number_z,delta_z)
        
    #imort matrix A and vector C coresponding to boundary
    A=boundary_type.A(number_z,delta_z)       
    C=boundary_type.C(number_z,delta_z)
    
    #import source
    F=source_typ.f0(number_t,number_z,delta_t,delta_z)
    
    #Solving the linear equation sistem for each time step
    for i in np.arange(number_t-1):
        T[i+1]=np.dot(A,T[i])+C+F[i]
    
    #tranpose solution grid
    T=np.transpose(T)
    
    #plot
    plt.imshow(T,cmap="jet",aspect='auto')   
    z_ticks_stepsize=0.1
    plt.yticks(np.arange((max_z)/z_ticks_stepsize)*number_z/(max_z/z_ticks_stepsize),np.round((np.array(-np.arange((max_z/z_ticks_stepsize)))*z_ticks_stepsize)+max_z*0.5,2))
    t_ticks_stepsize = 60
    plt.xticks(np.arange((max_t)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange((max_t/t_ticks_stepsize)))
    plt.xlabel('time [min]')
    plt.ylabel('position [m]')
    plt.colorbar().set_label('temperature [K]')
    plt.show()  
  
    #plot analytical solution
    if source == no and initial == normal and boundary == dirichlet and const_dirichlet == 0:
        plt.imshow(analytical_dirichlet(delta_z,delta_t,number_z,number_t),cmap="jet",aspect='auto')   
        z_ticks_stepsize=0.1
        plt.yticks(np.arange((max_z)/z_ticks_stepsize)*number_z/(max_z/z_ticks_stepsize),np.round((np.array(-np.arange((max_z/z_ticks_stepsize)))*z_ticks_stepsize)+max_z*0.5,2))
        t_ticks_stepsize = 60
        plt.xticks(np.arange((max_t)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange((max_t/t_ticks_stepsize)))
        plt.xlabel('time [min]')
        plt.ylabel('position [m]')
        plt.title('Analytical Dirichlet')
        plt.colorbar().set_label('temperature [K]')
        plt.show()      
    
    return T
    

    


number_t = 10000
max_t = 60*5       #[s]
max_z = 0.21         # [m]     

boundary = dirichlet  #cauchy, neumann, dirichlet, periodic
initial = const     #normal, const
source = moving   #fix,moving , no

run(number_t,max_t,max_z,boundary,initial,source)

