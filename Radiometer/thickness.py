import numpy as np
import matplotlib.pyplot as plt


n = np.arange(1,4,0.25)
angles = 90 - np.arccos(1/n)/np.pi*180
n = 1 / np.cos((90 - np.array(angles)) / 180 * np.pi)
print(angles)
print(n)

plt.plot(n,angles)
plt.show()