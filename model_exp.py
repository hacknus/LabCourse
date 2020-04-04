import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time
from Boundary import zenit, zenit_value
from Media import Soil, Granite, Ice


class Simulation:

	def __init__(self,medium,lat = 0, x = 10, dx = 0.014, t = 7, dt = 0.001):
		self.x = x
		self.dx = dx #m
		self.t = t #1*360 #days
		self.dt = dt #1/24/3 #days
		self.t_steps = int(self.t/self.dt)
		self.steps = int(self.x/self.dx)
		self.lat = lat
		self.medium = medium

		self.lower_boundary = 10 + 273.15 #degrees
		
	def stability(self):
		r = 0.5 #stability condition
		K = self.medium.lam/(self.medium.rho*self.medium.c_p)
		self.dx = np.sqrt(K*self.dt*60*60*24/r)
		self.steps = int(self.x // self.dx)
		self.dx = self.x / self.steps 
		print("x steps: ", self.steps, "dx: ",self.dx)
		print("t steps: ", self.t_steps, "dt: ",self.dt)

	def get_boundary(self,t,T):

		sigma = 5.67e-8
		E_0 = 1367

		zx = zenit_value(t,self.lat)
		P_sun = E_0 * (1 - self.medium.albedo) * np.cos(zx/180.0*np.pi) #+ 0.9 * E_0 * (self.medium.albedo) * np.cos(zx/180.0*np.pi)
		P_earth = self.medium.epsilon * sigma * T**4 #- 0.1 * self.medium.epsilon * sigma * T**4

		dT = (P_sun - P_earth) * self.dt*60*60*24 / (self.medium.c_p * self.medium.rho * self.dx) 

		#print("dT: ",dT)

		return  dT 


	def run(self,subtract = False):


		K = self.medium.lam / (self.medium.rho * self.medium.c_p)
		#K = 2.9/(890*2750)
		self.stability()

		grid = np.zeros((self.t_steps,self.steps)) + self.lower_boundary

		grid[:,0] = grid[:,0] + self.get_boundary(0,grid[:,0])
		#grid[:,0] = 273.15
		#grid[:,-1] = self.lower_boundary + 273.15

		r = K*self.dt*60*60*24/self.dx**2
		print("r: ",r)
		print("surface: ",grid[:,0])

		for t in range(1,grid.shape[0]):
			#print(t,"/",grid.shape[0])


			grid[t,0] = grid[t-1,0] + self.get_boundary(t*self.dt,grid[t-1,0])
			grid[t,-1] = self.lower_boundary

			grid[t,1:-1] = grid[t-1,1:-1] + r*(grid[t-1,2:] - 2*grid[t-1,1:-1] + grid[t-1,:-2])
			
			#print(T_loss)
			if subtract:

				T_loss = r*(grid[t-1,-2] - self.lower_boundary)
				diff = grid[t,1:-2] - grid[t-1,1:-2]
				#diff = grid[t,:-2] - grid[t,1:-1]
				#print(diff)
				print("Sum of Delta T_i: ",np.sum(diff))
				if t > 10:
					pass
					#exit()
				grid[t,0] = grid[t,0] - T_loss - np.sum(diff)

		xscale = 10
		yscale = 1
		#plt.imshow(grid[::xscale,::yscale].T)
		#plt.xticks(np.arange(0,self.t_steps/xscale,step=100), [ int(i*self.dt*xscale) for i in np.arange(0,self.t_steps/xscale,step=100)])
		#plt.yticks(np.arange(0,self.steps/yscale,step=100), [ "{:.1f}".format(i*self.dx*yscale) for i in np.arange(0,self.steps/yscale,step=100)])
		#plt.ylabel("depth [m]")
		#plt.xlabel("days [d]")
		#plt.colorbar()
		#plt.tight_layout()
		#plt.show()

		plt.plot(np.linspace(0,self.t,self.t_steps),grid[:,0])
		#plt.show()

	def get_true_data(self):

		filename = "bodemessnetz_datenabfrage.csv"

		df = pd.read_csv(filename,encoding = "ISO-8859-1")

		plt.title("Zollikofen-Oberacker")
		air_temp = df["Lufttemperatur 2 m"].astype(float) + 273.15
		soil_temp = df["Bodentemperatur 20 cm"].astype(float) + 273.15
		plt.plot(range(len(air_temp)),air_temp,label="airtemp")
		plt.plot(range(len(soil_temp)),soil_temp,label="soiltemp")
		plt.xlabel("days")
		plt.ylabel("T [K]")
		plt.legend()
		plt.show()
		exit()


if __name__ == "__main__":

	Explicit = Simulation(Granite)
	#Explicit.get_true_data()
	Explicit.run()
	Explicit.run(True)
	plt.show()
