import numpy as np
import matplotlib.pyplot as plt


theta = np.linspace(-np.pi/2,np.pi/2,1000)
phi = 0
l = 0.01362692990909
a = 3*l
b = 3*l
k = 2*np.pi/l

A = np.abs(np.sinc(a*k*np.sin(theta)*np.cos(phi)/(2*np.pi))*np.sinc(b*k*np.sin(theta)*np.sin(phi)/(2*np.pi)))*np.sqrt(np.cos(theta)**2*np.cos(phi)**2 + np.sin(phi)**2)
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(theta, A, color="black")
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_theta_zero_location("N")
plt.tight_layout()
plt.savefig("pattern.pdf")
plt.show()