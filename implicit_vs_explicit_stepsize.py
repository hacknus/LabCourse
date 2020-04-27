import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import time
from matplotlib.pyplot import figure
import numpy as np
from numpy.linalg import inv
from inversion import invert
from Boundary import zenit_value
from matrix import Trapezoidal, Trapezoidal_Iso, Leap_Frog, Leap_Frog_Iso 
from Media import Granite, Ice, Soil
from model_exp import Simulation
from model_imp import Implicit


#parameters for the function
max_t=10         #[d] time periode 
max_z=10           #[m] depth
delta_z=0.1         #[m] depth step size 
delta_t=0.00001                    #delta_t<=0.04884 falls delta_z=0.1
latitude=46.95       #[deg] latitude   
medium=Granite        #(Granite, Ice or Soil) media 
methode=Leap_Frog      #(Trapezoidal,Trapezoidal_Iso, Leap_Frog or Lea_frog_Iso) method 

#fixed parameter
S_0=1367                #solar constant [W/m^2]
sigma=5.67*1e-8         #Stefan-Bolzmann cosntant [W/m^2 K^-4]
T_res=10+273.15         #reservoir Temperature [K]
#delta_t=1/(2*medium.lam*24*3600/(medium.c_p*medium.rho*delta_z**2))
print(delta_t)

explicit = np.transpose(Simulation(medium, lat = latitude, x = max_z, dx = delta_z, t = max_t, dt = 0.00001).run(subtract =True ,stability =False))
implicit_1 = Implicit.run(max_t, 0.00001, max_z, delta_z, latitude, medium, Leap_Frog)
implicit_2 = Implicit.run(max_t, 0.00001, max_z, delta_z, latitude, medium, Trapezoidal)
explicit_akt = np.transpose(Simulation(medium, lat = latitude, x = max_z, dx = delta_z, t = max_t, dt = delta_t).run(subtract =True ,stability =False))
implicit_1_akt = Implicit.run(max_t, delta_t, max_z, delta_z, latitude, medium, Leap_Frog)
implicit_2_akt4 = Implicit.run(max_t, 0.0001, max_z, delta_z, latitude, medium, Trapezoidal)
implicit_2_akt3 = Implicit.run(max_t, 0.001, max_z, delta_z, latitude, medium, Trapezoidal)
implicit_2_akt2 = Implicit.run(max_t, 0.01, max_z, delta_z, latitude, medium, Trapezoidal)
implicit_2_akt1 = Implicit.run(max_t, 0.05, max_z, delta_z, latitude, medium, Trapezoidal)


#choose plot sequenz
max_z_show=1           #[m] plot sequenz depth
max_t_show=max_t            #[d] plot end time sequenz
min_t_show=0              #[d] plot start time sequenz
t_ticks_stepsize=1          #[d]

#variables to make code clearer
number_t=int(max_t/delta_t)
number_z=int(max_z/delta_z)
max_z_show_steps=int(max_z_show/max_z*number_z)
max_t_show_steps=int(max_t_show/max_t*number_t)
min_t_show_steps=int(min_t_show/max_t*number_t)

norm = plt.Normalize(0,2.5) 
i=int(delta_t*100000)
print(i)

#compare different methodes
#exp vs trap
plt.imshow(np.abs(implicit_2-explicit), cmap="jet",aspect='auto')
plt.yticks(np.arange(max_z_show+1)*number_z/max_z,np.arange(max_z_show+1))
plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('temperature [K]')
plt.savefig('resolution/trap_vs_exp_0.00001.pdf'.format(delta_t))
plt.show()

#imp vs trap
plt.imshow(np.abs(implicit_2-implicit_1), cmap="jet",aspect='auto')
plt.yticks(np.arange(max_z_show+1)*number_z/max_z,np.arange(max_z_show+1))
plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('temperature [K]')
plt.savefig('resolution/trap_vs_leap_0.00001.pdf'.format(delta_t))
plt.show()




#uppermost layer
plt.plot(np.abs(np.transpose(np.transpose(implicit_2)[::5000])-implicit_2_akt1)[0],label='0.05 d')
plt.plot(np.abs(np.transpose(np.transpose(implicit_2)[::1000])-implicit_2_akt2)[0][::5],label='0.01 d') 
plt.plot(np.abs(np.transpose(np.transpose(implicit_2)[::100])-implicit_2_akt3)[0][::50],label='0.001 d')
plt.plot(np.abs(np.transpose(np.transpose(implicit_2)[::10])-implicit_2_akt4)[0][::500],label='0.0001 d')
plt.xticks(np.arange((max_t_show+1-min_t_show))*20,np.arange(min_t_show, (max_t_show+1)))
plt.xlabel('time [d]')
plt.legend(loc='upper right')
plt.ylabel('temperature difference  [K]')
plt.yscale('log')
plt.savefig('resolution/uppermost_log.pdf')
plt.show()




#compare different stepsizes
plt.imshow(np.abs(np.transpose(np.transpose(explicit)[::i])-explicit_akt), cmap="jet",aspect='auto')
plt.yticks(np.arange(max_z_show+1)*number_z/max_z,np.arange(max_z_show+1))
plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('temperature [K]')#plt.savefig('resolution/exp{}_vs_explicit0.00001.pdf'.format(delta_t))
plt.show()

#plt.imshow(np.abs(np.transpose(np.transpose(implicit_1)[::i])-implicit_1_akt),cmap="jet",aspect='auto',norm=norm)
plt.yticks(np.arange(max_z_show+1)*number_z/max_z,np.arange(max_z_show+1))
plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('temperature [K]')
plt.savefig('resolution/leap-frog{}_vs_leap-frog_0.00001.pdf'.format(delta_t))
plt.show()

#plt.imshow(np.abs(np.transpose(np.transpose(implicit_2)[::i])-implicit_2_akt),cmap="jet",aspect='auto',norm=norm)
plt.yticks(np.arange(max_z_show+1)*number_z/max_z,np.arange(max_z_show+1))
plt.xticks(np.arange((max_t_show+1-min_t_show)/t_ticks_stepsize)*number_t/(max_t/t_ticks_stepsize),np.arange(min_t_show, (max_t_show+1)*t_ticks_stepsize,t_ticks_stepsize))
plt.xlabel('time [d]')
plt.ylabel('depth [m]')
plt.colorbar().set_label('temperature [K]')
plt.savefig('resolution/trapezoidal{}_vs_trapezoidal_0.00001.pdf'.format(delta_t))
plt.show()

