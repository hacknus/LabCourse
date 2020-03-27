import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time
from zentit2 import zenit



class Simulation:

	def __init__(self):
		self.x = 10
		self.dx = 0.01 #m
		self.t = 24*100 #days
		self.dt = 60*60 #s

		self.lower_boundary = 15 #degrees
		
		#soil
		self.a = 0.17 
		self.epsilon = 0.92 
		self.rho = 2.04
		self.c_p = 1.84
		self.lam = 0.52

	def get_boundary(self,t,T):

		sigma = 5.67
		E_0 = 1367

		zx = zenit(self.t,self.dt)[t]
		P_sun = E_0 * (1 - self.a) * np.cos(zx)
		P_earth = self.epsilon * sigma * T**4

		dT = (P_sun - P_earth) * self.dt / (self.c_p * self.rho * self.dx)

		return  dT 


	def run(self):

		grid = np.zeros((self.t,self.x)) + 273.15

		grid[:,0] = self.get_boundary(0,0)
		grid[:,-1] = self.lower_boundary + 273.15

		K = self.lam/(self.rho*self.c_p)/1000
		#K = 2.9/(890*2750)
		r = K*self.dt/self.dx**2

		print(r)

		for t in range(1,self.t):
			for x in range(1,self.x-1):
				grid[t,x] = grid[t-1,x] + r*(grid[t-1,x+1] - 2*grid[t-1,x] + grid[t-1,x-1])
			
			# print(grid[t-1,x])
			# print(grid[t-1,x+1])
			# print(- 2*grid[t-1,x])
			# print(grid[t-1,x-1])
			# print(t,x,r,grid[t,x])

			#grid[t+1,0] = grid[t,0] #+ self.get_boundary(t,grid[t,x])
			#grid[t+1,-1] = self.lower_boundary

		plt.imshow(grid[::300,-1:0:-1].T)
		plt.colorbar()
		plt.show()





if __name__ == "__main__":

	Explicit = Simulation()
	Explicit.run()
