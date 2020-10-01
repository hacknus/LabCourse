import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time
from Boundary import zenit, zenit_value
from Media import Soil, Granite, Ice
from model_an import AnalyticalSim

class Simulation:

	def __init__(self,medium,lat = 0, x = 10, dx = 0.014, t = 360, dt = 0.001):
		self.x = x
		self.dx = dx #m
		self.t = t #1*360 #days
		self.dt = dt #1/24/3 #days
		self.t_steps = int(self.t/self.dt)
		self.steps = int(self.x/self.dx)
		self.lat = 46
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
		P_sun = E_0 * (1 - self.medium.albedo) * np.cos(zx/180.0*np.pi) 
		P_earth = self.medium.epsilon * sigma * T**4 #* 0.42# - 0.2 * self.medium.epsilon * sigma * T**4

		dT = (P_sun - P_earth) * self.dt*60*60*24 / (self.medium.c_p * 1 * self.medium.rho * self.dx) 

		#print("dT: ",dT)

		#plt.scatter(t,dT,color="black")

		return  dT 


	def run(self,subtract = False, stability = True):

		t0 = 121

		K = self.medium.lam / (self.medium.rho * self.medium.c_p)
		#K = 2.9/(890*2750)
		if stability:
			self.stability()

		grid = np.zeros((self.t_steps,self.steps)) + self.lower_boundary

		grid[:,0] = grid[:,0] + self.get_boundary(t0 + self.dt,grid[0,0])
		#grid[:,0] = 273.15
		#grid[:,-1] = self.lower_boundary + 273.15

		r = K*self.dt*60*60*24/self.dx**2
		print("r: ",r)
		print("surface: ",grid[:,0])

		for t in range(1,grid.shape[0]):
			#print(t,"/",grid.shape[0])


			grid[t,0] = grid[t-1,0] + self.get_boundary(t*self.dt + t0,grid[t-1,0])
			grid[t,-1] = self.lower_boundary

			grid[t,1:-1] = grid[t-1,1:-1] + r*(grid[t-1,2:] - 2*grid[t-1,1:-1] + grid[t-1,:-2])
			
			#print(T_loss)
			if subtract:

				# if grid[t,1] > grid[t,0]:
				#   	grid[t,0] = grid[t,0] + r*(-grid[t,0] + grid[t,1])

				T_loss = r*(grid[t-1,-2] - self.lower_boundary)
				diff = grid[t,1:-2] - grid[t-1,1:-2]
				#diff = grid[t,:-2] - grid[t,1:-1]
				#print(diff)
				#print("Sum of Delta T_i: ",np.sum(diff))
				if t > 10:
					pass
					#exit()
				grid[t,0] = grid[t,0] - T_loss - np.sum(diff)


		return grid

	def plot(self,grid):
		xscale = 2
		yscale = 1

		norm = plt.Normalize(273,330)

		plt.subplot(211)
		plt.imshow(grid[::xscale,::yscale].T,cmap="jet",aspect="auto",norm=norm)
		plt.xticks(np.arange(0,self.t_steps/xscale,step=50), [ int(i*self.dt*xscale) for i in np.arange(0,self.t_steps/xscale,step=50)])
		plt.yticks(np.arange(0,self.steps/yscale,step=10), [ "{:.1f}".format(i*self.dx*yscale) for i in np.arange(0,self.steps/yscale,step=10)])
		plt.title("explicit")
		plt.ylabel("depth [m]")
		plt.xlabel("days [d]")
		plt.colorbar()
		plt.tight_layout()

		K = self.medium.lam / (self.medium.rho * self.medium.c_p)

		t = np.array(range(0,grid.shape[0]))*self.dt
		z = np.array(range(0,grid.shape[1]))*self.dx

		Analytic = AnalyticalSim(K,t,z,self.x,self.lower_boundary)


		plt.subplot(212)
		Analytic.fit_omega(grid[:,0],t,grid.shape[0]//5*4)
		an_grid = Analytic.plot(plt,t,z)

		plt.imshow(an_grid[::xscale,::yscale].T,cmap="jet",aspect="auto",norm=norm)
		plt.xticks(np.arange(0,self.t_steps/xscale,step=50), [ int(i*self.dt*xscale) for i in np.arange(0,self.t_steps/xscale,step=50)])
		plt.yticks(np.arange(0,self.steps/yscale,step=10), [ "{:.1f}".format(i*self.dx*yscale) for i in np.arange(0,self.steps/yscale,step=10)])
		plt.title("analytical")
		plt.ylabel("depth [m]")
		plt.xlabel("days [d]")
		plt.colorbar()
		plt.tight_layout()
		"""

		plt.title("explicit")
		plt.ylabel("temperature [C]")
		plt.xlabel("days [d]")
		plt.plot(np.linspace(0,self.t,self.t_steps),grid[:,int(0.20/self.dx)])
		plt.plot(np.linspace(0,self.t,self.t_steps),grid[:,int(0.35/self.dx)])
		plt.plot(np.linspace(0,self.t,self.t_steps),grid[:,int(0.60/self.dx)])

		"""
		#self.get_true_data()
		plt.savefig("test2.pdf")
		plt.show()

		i = -20
		sl = t[i]
		plt.title("slide at t = {:.1f} d".format(sl))

		plt.plot(z,grid[i,:],label="explicit")
		#plt.xticks(np.arange(0,self.steps/yscale,step=10), [ "{:.1f}".format(i*self.dx*yscale) for i in np.arange(0,self.steps/yscale,step=10)])
		plt.xlabel("depth [m]")
		plt.ylabel(r"$T$ [$\degree$C")

		plt.plot(z,an_grid[i,:],label="analytical")

		plt.legend()

		plt.show()

	def get_true_data(self):

		filename = "bodemessnetz_datenabfrage.csv"

		df = pd.read_csv(filename,encoding = "ISO-8859-1")

		plt.title("Zollikofen-Oberacker")
		air_temp = df["Lufttemperatur 2 m"].astype(float) + 273.15
		soil_temp1 = df["Bodentemperatur 20 cm"].astype(float) + 273.15
		soil_temp2 = df["Bodentemperatur 35 cm"].astype(float) + 273.15
		soil_temp3 = df["Bodentemperatur 60 cm"].astype(float) + 273.15
		plt.plot(range(len(air_temp)),air_temp,label="airtemp")
		plt.plot(range(len(soil_temp1)),soil_temp1,label="soiltemp 20")
		plt.plot(range(len(soil_temp2)),soil_temp2,label="soiltemp 35")
		plt.plot(range(len(soil_temp3)),soil_temp3,label="soiltemp 60")
		plt.xlabel("days")
		plt.ylabel("T [K]")



if __name__ == "__main__":

	Explicit = Simulation(Soil)
	Explicit = Simulation(Granite,lat = 46, x = 2, dx = 0.001, t = 10, dt = 0.5/72)
	grid = Explicit.run(True)

	#plt.show()
	Explicit.plot(grid)

	#plt.savefig("test.pdf")
	#plt.show()
