import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import time
from matplotlib.pyplot import figure
import numpy as np
from numpy.linalg import inv
from inversion import invert
from Boundary import zenit_value
from ComputerSim.matrix import Trapezoidal, Trapezoidal_Iso, Leap_Frog, Leap_Frog_Iso
from Media import Granite, Ice, Soil



#parameters for the function
max_t=365*2+1              #[d] time periode 
delta_t=0.01        #[d] time step size
max_z=10           #[m] depth
delta_z=0.01          #[m] depth step size
latitude=0          #[deg] latitude   
medium=Granite          #(Granite, Ice or Soil) media 
methode=Trapezoidal      #(Trapezoidal,Trapezoidal_Iso, Leap_Frog or Lea_frog_Iso) method 

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
        
        #imort the matching matrices A, B and vector C
        A=methode.A(number_z+1,r)  
        invB=invert(methode.B(number_z+1,r))        #np.linalg.inv(methode.B(number_z+1,r))             
        C=methode.C(number_z+1,r,T_res)
        #Solving the linear equation sistem for each time step
        for i in np.arange(number_t-1):
            #amplify boundary conditions for each time step
            C[0]=-delta_t*24*3600/(medium.c_p*medium.rho*delta_z)*(-medium.epsilon*sigma*T[i][0]**4+(1-medium.albedo)*S_0*np.cos(zenit_value(i*delta_t,latitude)/360*2*np.pi))
            T[i+1][:]=np.dot(invB,np.dot(A,T[i][:])-C)
        #tranpose solution grid and delete T_loss line
        T=np.delete(T.transpose(),number_z,0)
        return T

if __name__ == "__main__":

    #choose plot sequenz
    max_z_show=max_z            #[m] plot sequenz depth
    max_t_show=max_t            #[d] plot end time sequenz
    min_t_show=365               #[d] plot start time sequenz
    t_ticks_stepsize=30.5          #[d]
    
    #variables to make code clearer
    number_t=int(max_t/delta_t)
    number_z=int(max_z/delta_z)
    max_z_show_steps=int(max_z_show/max_z*number_z)
    max_t_show_steps=int(max_t_show/max_t*number_t)
    min_t_show_steps=int(min_t_show/max_t*number_t)
 
    #create matix sequenz
    tstart=time.time() 
    R0=Implicit.run(max_t,delta_t,max_z,delta_z,latitude,medium,methode).T
    tend=time.time()
    print(tstart-tend)
    R0=R0[min_t_show_steps:max_t_show_steps+1].T
    
    #plot matrix sequenz
    norm = plt.Normalize(220,330)    
    plt.imshow(R0[0:max_z_show_steps+1],cmap="jet",aspect='auto',norm=norm)
    plt.yticks(np.arange(max_z_show+1)*number_z/max_z,np.arange(max_z_show+1))
    #plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
    plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),['1/1', '2/1', '3/1', '4/1', '5/1', '6/1', '7/1', '8/1', '9/1', '10/1', '11/1', '12/1', '1/1'])
    #plt.title('Temperature at {} deg Latitude'.format(latitude))
    #plt.xlabel('time [d]')
    plt.xlabel('date [month/day]')
    plt.ylabel('depth [m]')

    plt.colorbar().set_label('temperature [K]')
    
    plt.savefig("latitude/depthprofile/depth_profile_zenith_{}.pdf".format(latitude))
    plt.show()   
    

    
    #plot uppermost layer
    plt.plot(R0[0])
    #plt.title('Surface Temperature at {} deg Latitude'.format(latitude))
    #plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
    plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),['1/1', '2/1', '3/1', '4/1', '5/1', '6/1', '7/1', '8/1', '9/1', '10/1', '11/1', '12/1', '1/1'])
    #plt.xlabel('time [d]')
    plt.xlabel('date [month/day]')
    plt.ylabel('temperature [K]')
    plt.savefig("latitude/uppermost/uppermost_{}.pdf".format(latitude))
    plt.show()
    
    #plot zenith angle
    omega0=np.zeros(number_t)
    for i in np.arange(number_t):
        omega0[i]=zenit_value(i*delta_t,latitude)
    omega=omega0[min_t_show_steps:max_t_show_steps]
    plt.plot(omega)
    #plt.title('Zenith Angle at {} deg Latitude'.format(latitude))
    #plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
    plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),['1/1', '2/1', '3/1', '4/1', '5/1', '6/1', '7/1', '8/1', '9/1', '10/1', '11/1', '12/1', '1/1'])
    #plt.xlabel('time [d]')
    plt.xlabel('date [month/day]')
    plt.ylabel('zenith angle [deg]')
    plt.savefig("latitude/zenith/zenith_{}.pdf".format(latitude))
    plt.show()
    

    