import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.integrate as integrate
import math
from numpy.linalg import inv

def zenit(steps_t = 800*24,delta_t = 60*60):

	year = 365.2422*24*60*60 	#[s]
	sid_day = 24*60*60/(1+1/365.2422)                   #23*60*60+56*60+4.1 #[s]
	b = 23.43666/360*2*np.pi 	#ecliptic
	c = 69/360*2*np.pi 			#latitude
	omega = np.zeros(steps_t)

	for i in np.arange(steps_t):
		t = i*delta_t 
		d = t/sid_day*2*np.pi #earth rotation angle
		a = t/year*2*np.pi #sun rotation angle
		zx = np.arccos(np.cos(a)*np.cos(c)*np.cos(d)+np.sin(a)*np.cos(b)*np.cos(c)*np.sin(d)+np.sin(a)*np.sin(b)*np.sin(c))/(2*np.pi)*360
		#print(i)
		#print(zx)
		if zx <= 90:
			omega[i] = zx
		else:
			omega[i] = 90
	return omega

def zenit_value(t,c0=69):
	year = 365.2422*24*60*60 	       #year[s]
	sid_day = 24*60*60/(1+1/365.2422)  #sideral day [s]
	b = 23.43666/360*2*np.pi           #ecliptic [rad]
	c = c0/360*2*np.pi 			       #latitude [rad]


	t = t*60*60*24+(10*24*60*60+37*60)                                     #change [s] in [d]  #start at 00:00 1. january 2019
	d = t/sid_day*2*np.pi+(5*60+23)/(24*60)*2*np.pi-(15-7.457)/360*2*np.pi #earth rotation angle     #shortest day was at 23:23 21.12.2018# local correction-(15-7.457)  
	a = t/year*2*np.pi-np.pi/2                                             #sun rotation angle #start at shortest day    
	zx = np.arccos(np.cos(a)*np.cos(c)*np.cos(d)+np.sin(a)*np.cos(b)*np.cos(c)*np.sin(d)+np.sin(a)*np.sin(b)*np.sin(c))/(2*np.pi)*360
    
	if zx <= 90:
		omega = zx
	else:
		omega = 90
		
	return omega


if __name__ == "__main__":
	omega = zenit()
	print(min(omega))
	plt.plot(omega)


        