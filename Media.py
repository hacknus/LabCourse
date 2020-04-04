
class Medium:

	def __init__(self,name,rho,lam,epsilon,albedo,c_p):
		self.name = name
		self.rho = rho
		self.lam = lam
		self.epsilon = epsilon
		self.albedo = albedo
		self.c_p = c_p


Soil = Medium("soil",2.04e3,0.52,0.92,0.17,1.84e3)
Granite = Medium("granite",2.75e3,2.9,0.45,0.3,0.89e3)
Ice = Medium("ice",0.917e3,2.25,0.95,0.5,2.04e3)

if __name__ == "__main__":

	def test_func_for_donat(t,dt,z,dz,medium):
		print("you have selected: ",medium.name)

	test_func_for_donat(100,1,10,1,Soil)
	test_func_for_donat(100,1,10,1,Granite)
	test_func_for_donat(100,1,10,1,Ice)