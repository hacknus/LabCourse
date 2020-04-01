import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time
from zentit2 import zenit



class Simulation:

	def __init__(self):
		self.x = 10
		self.dx = 1 #m
		self.t = 100 #1*360 #days
		self.dt = 1 #days
		self.steps = 100

		self.lower_boundary = 15 #degrees
		
		#soil
		self.a = 0.17 
		self.epsilon = 0.92 
		self.rho = 2.04e3
		self.c_p = 1.84e3
		self.lam = 0.52

	def stability(self):
		r = 0.5 #stability condition
		K = self.lam/(self.rho*self.c_p)
		self.dx = np.sqrt(K*self.dt*60*60*24/r)
		print(self.dx)
		self.steps = int(self.x // self.dx)
		self.dx = self.x / self.steps 

	def get_boundary(self,t,T):

		sigma = 5.67e-8
		E_0 = 1367

		zx = zenit(self.t,self.dt)[t]
		P_sun = E_0 * (1 - self.a) * np.cos(zx/180.0*np.pi)
		P_earth = self.epsilon * sigma * T**4

		dT = (P_sun - P_earth) * self.dt*60*60*24 / (self.c_p * self.rho * self.dx)

		#print("dT: ",dT)

		return  dT 


	def run(self):


		K = self.lam/(self.rho*self.c_p)
		#K = 2.9/(890*2750)
		self.stability()

		grid = np.zeros((self.t,self.steps)) + 273.15 + self.lower_boundary

		grid[:,0] = grid[:,0] + self.get_boundary(0,grid[:,0])
		#grid[:,0] = 273.15
		#grid[:,-1] = self.lower_boundary + 273.15

		r = K*self.dt*60*60*24/self.dx**2
		print("r: ",r)
		print("surface: ",grid[:,0])

		for t in range(1,grid.shape[0]):
			print(t,"/",grid.shape[0])
			for x in range(1,grid.shape[1]-1):
				#print("   ",x,"/",grid.shape[1])
				grid[t,0] = grid[t,0] + self.get_boundary(t,grid[t,0])
				grid[:,-1] = self.lower_boundary + 273.15
				grid[t,x] = grid[t-1,x] + r*(grid[t-1,x+1] - 2*grid[t-1,x] + grid[t-1,x-1])
			
			# print(grid[t-1,x])
			# print(grid[t-1,x+1])
			# print(- 2*grid[t-1,x])
			# print(grid[t-1,x-1])
			# print(t,x,r,grid[t,x])

			#grid[t+1,0] = grid[t,0] #+ self.get_boundary(t,grid[t,x])
			#grid[t+1,-1] = self.lower_boundary

		#plt.imshow(grid[::1,-1:0:-5].T)
		plt.imshow(grid[::1,::1].T)
		plt.colorbar()
		plt.tight_layout()
		plt.show()





if __name__ == "__main__":

	Explicit = Simulation()
	Explicit.run()
