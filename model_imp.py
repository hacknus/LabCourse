import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from numpy.linalg import inv
from Boundary import zenit_value
from matrix import Trapezoidal, Trapezoidal_Iso, Leap_Frog, Leap_Frog_Iso 
from Media import Granite, Ice, Soil

#parameters for the function
max_t=7                 #[d] time periode 
delta_t=0.001           #[d] time step size
max_z=10                #[m] depth
delta_z=0.014           #[m] depth step size
latitude=0              #[deg] latitude 
medium=Granite          #(Granite, Ice or Soil) media 
methode=Trapezoidal_Iso #(Trapezoidal,Trapezoidal_Iso, Leap_Frog or Lea_frog_Iso) method 

#fixed parameter
S_0=1367                #solar constant [W/m^2]
sigma=5.67*1e-8         #Stefan-Bolzmann cosntant [W/m^2 K^-4]
T_res=10+273.15         #reservoir Temperature [K]




class Implicit:

    def run(max_t, delta_t, max_z, delta_z, latitude, medium, methode):
        
        #define variables to make the code clearer
        number_t=int(max_t/delta_t)
        number_z=int(max_z/delta_z)
        r=medium.lam*(delta_t*24*3600)/(medium.c_p*medium.rho*delta_z**2)
        
        #solution grid
        T=np.zeros((number_t,number_z+1))
        T[0][:]=T_res
        print(T[0][:])
        
        #imort the matching matrices A, B and vector C
        A=methode.A(number_z+1,r) 
        invB=np.linalg.inv(methode.B(number_z+1,r))
        C=methode.C(number_z+1,r,T_res)
        
        #Solving the linear equation sistem for each time step
        for i in np.arange(number_t-1):
            #amplify boundary conditions for each time step
            C[0]=-delta_t*24*3600/(medium.c_p*medium.rho*delta_z)*(-medium.epsilon*sigma*T[i][0]**4+(1-medium.albedo)*S_0*np.cos(zenit_value(i*delta_t,latitude)/360*2*np.pi))
            T[i+1][:]=np.dot(invB,np.dot(A,T[i][:])-C)
        #tranpose solution grid and delete T_loss line
        T=np.delete(T.transpose(),number_z-1,0)
        return T

if __name__ == "__main__":

    max_z_show=1        #[m] plot sequenz depth
    max_t_show=max_t    #[t] plot end time sequenz
    min_t_show=0        #[t] plot start time sequenz
    scale_z=1           #scale factor z
    
    number_t=int(max_t/delta_t)
    number_z=int(max_z/delta_z)
    max_z_show_steps=int(max_z_show/max_z*number_z)
    max_t_show_steps=int(max_t_show/max_t*number_t)
    min_t_show_steps=int(min_t_show/max_t*number_t)
    print(max_t_show_steps)


    R0=Implicit.run(max_t,delta_t,max_z,delta_z,latitude,medium,methode)
    print(R0.shape)
    R=np.zeros((max_z_show_steps*scale_z,max_t_show_steps-min_t_show_steps))
    for j in np.arange(max_z_show_steps):
        for l in np.arange(scale_z):
            for i in np.arange(max_t_show_steps-min_t_show_steps):
                R[j*scale_z+l][i]=R0[j][i+min_t_show_steps]
            
        
    plt.matshow(R0)
    plt.title('Trapezoidal')
    plt.yticks(np.arange(max_z_show+1)*number_z/max_z*scale_z,np.arange(max_z_show+1))
    plt.xticks(np.arange(max_t_show-min_t_show+1)*number_t/max_t,np.arange(min_t_show,max_t_show+1))
    plt.xlabel('time [d]')
    plt.ylabel('depth [m]')
    plt.colorbar().set_label('Temperature [K]')
    plt.show()        

    plt.plot(R0[0], label= 'Trapezoidal')
    #plt.plot(T_trap, label='Trapezoidal')
    plt.title('Surface Temperature $T_{0}$')
    plt.xticks(np.arange(max_t+1)*number_t/max_t,np.arange(max_t+1))
    plt.xlabel('time [d]')
    plt.ylabel('Temperature [K]')
    plt.legend()
    plt.show()
 