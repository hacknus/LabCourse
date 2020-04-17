import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time
from Boundary import zenit, zenit_value
from Media import Soil, Granite, Ice


class AnalyticalSim:

	def __init__(self,K,t,z,zmax,TC):
		self.T0 = 200
		self.T = 2
		self.omega = 2
		self.phase = 2
		self.K = K
		self.t = t
		self.z = z
		self.zmax = zmax
		self.Tcore = TC

	def func(self,t,omega,T,T0,phase):
		return T*np.cos(t*omega - phase) + T0
		
	def fit_omega(self, T, t, window_low = 0, window_up = -1):

		p0 = [5,20,300,2]
		#plt.plot(t,self.func(t,*p0))
		popt,_ = curve_fit(self.func,t[window_low:window_up],T[window_low:window_up],p0=p0)

		self.phase = popt[3]
		self.T0 = popt[2]
		self.T = popt[1]
		self.omega = popt[0]

		#self.T0 = 10 +  273.15
		print(popt)
		# plt.show()
		# plt.plot(t,T,label="implicit boundary")
		# plt.plot(t,self.func(t,*popt),ls="--",label="fit")
		# plt.ylabel(r'$T$ [K]')
		# plt.xlabel(r'$t$ [d]')
		# plt.legend()
		# plt.savefig("fit.pdf",dpi=300)
		# plt.show()


	def time_diff(self,z):
		return z / np.sqrt(2 * self.omega/(60*60*24) * self.K)

	def dampening(self,z):
		#print(np.exp( -np.sqrt(self.omega/(60*60*2) / (2 * self.K)) * z),-np.sqrt(self.omega / (2 * self.K)) * z,z)
		return np.exp( -np.sqrt(self.omega/(60*60*24) / (2 * self.K)) * (z) )

	def T_func(self,t,z):
		T0 = (-self.T - self.T0 + self.Tcore) * self.dampening(z) * np.cos(self.omega * t - np.sqrt(self.omega / (2*60*60*24 * self.K)) * (z) - self.phase)
		return T0 + (self.Tcore + (self.T0 - self.Tcore) * (self.zmax - z) / self.zmax)

	def plot(self,plt_in,t,z):
		print(len(t))
		print(len(z))
		#print(self.T_func(t,z).shape)
		# grid = np.zeros((len(t),len(z)))
		# for ti in range(len(t)):
		# 	for zi in range(len(z)):
		# 		grid[ti,zi] = self.T_func(self.t[ti],self.z[zi])
		t = self.t[:,None]
		z = self.z[None,:]

		z = np.zeros((len(self.t),len(self.z)) )
		z[:,:] = self.z

		grid = self.T_func(t,z)
		return grid



if __name__ == "__main__":

	pass
