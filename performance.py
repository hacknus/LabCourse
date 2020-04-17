import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time
from Boundary import zenit, zenit_value
from Media import Soil, Granite, Ice
from matrix import Trapezoidal, Trapezoidal_Iso, Leap_Frog, Leap_Frog_Iso 

from model_exp import Simulation
from model_imp import Implicit


#parameters for the function
max_t = 7                 #[d] time periode 
delta_t = 0.001           #[d] time step size
max_z = 10                #[m] depth
delta_z = 0.014           #[m] depth step size
latitude = 0              #[deg] latitude 
medium = Granite          #(Granite, Ice or Soil) media 
methode = Leap_Frog_Iso   #(Trapezoidal,Trapezoidal_Iso, Leap_Frog or Lea_frog_Iso) method 

#fixed parameter
S_0 = 1367                #solar constant [W/m^2]
sigma = 5.67*1e-8         #Stefan-Bolzmann cosntant [W/m^2 K^-4]
T_res = 10+273.15         #reservoir Temperature [K]



#choose plot sequenz
max_z_show = 1        #[m] plot sequenz depth
max_t_show = max_t    #[t] plot end time sequenz
min_t_show = 0        #[t] plot start time sequenz
scale_z = 1           #scale factor z

#variables to make code clearer
number_t = int(max_t / delta_t)
number_z = int(max_z / delta_z)
max_z_show_steps = int(max_z_show / max_z * number_z)
max_t_show_steps = int(max_t_show / max_t * number_t)
min_t_show_steps = int(min_t_show / max_t * number_t)


sum_num = 5

implicit_dt = [[] for i in range(sum_num)]
explicit_dt = [[] for i in range(sum_num)]

z = np.arange(2,10.5,0.5)

for i in range(sum_num):
	for max_z in z:
		Explicit = Simulation(medium,lat = latitude, x = max_z, dx = delta_z, t = max_t, dt = delta_t)

		implicit_t0 = time.time()

		R0 = Implicit.run(max_t,delta_t,max_z,delta_z,latitude,medium,methode)

		implicit_dt[i].append(time.time() - implicit_t0)


		explicit_t0 = time.time()

		grid = Explicit.run()

		explicit_dt[i].append(time.time() - explicit_t0)

		print(" max_z = ", max_z)
		print(" Explicit: {:.4f} s".format(explicit_dt[i][-1]))
		print(" Implicit: {:.4f} s".format(implicit_dt[i][-1]))
	#plt.scatter(z,explicit_dt[i],marker="x",color="red")
	#plt.scatter(z,implicit_dt[i],marker="x",color="blue")
explicit_dt_mean = np.array(explicit_dt).mean(axis=0)
explicit_dt_std = np.array(explicit_dt).std(axis=0)
implicit_dt_mean = np.array(implicit_dt).mean(axis=0)
implicit_dt_std = np.array(implicit_dt).std(axis=0)
plt.errorbar(z,explicit_dt_mean,yerr=explicit_dt_std,fmt='.', markeredgecolor ="black",ecolor='black', capthick=2, color="red",capsize=2, elinewidth=1, markeredgewidth=0.5,ms=5,label="explicit")
plt.errorbar(z,implicit_dt_mean,yerr=implicit_dt_std,fmt='.', markeredgecolor ="black",ecolor='black', capthick=2, color="blue",capsize=2, elinewidth=1, markeredgewidth=0.5,ms=5,label="implicit")
#plt.plot(z,explicit_dt,label="explicit")
#plt.plot(z,implicit_dt,label="implicit")
plt.ylabel(r"$\overline{t}_{CPU}$ [s]")
plt.xlabel(r"$z_{max}$ [m]")
plt.legend()
plt.savefig("performance_z.pdf",dpi=300)
plt.show()


