import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from main import sky_temp


OS = platform.system()

                
for freq, c in zip([16, 17, 18, 19], ['red', 'blue', 'green', 'orange']):
    if OS == "Windows":
        K_and_C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(freq))  
        angles = pd.read_csv(r'{}GHZ\angles.csv'.format(freq))
    else:
        K_and_C = pd.read_csv('{}GHZ/K_and_C.csv'.format(freq))
        angles = pd.read_csv(r'{}GHZ/angles.csv'.format(freq))
    T_sky=np.zeros(10)
    T_sky_err=np.zeros(10)
    for i in np.arange(10):
        T_sky[i], T_sky_err[i] = sky_temp(angles,K_and_C,i)
    rel_thickness = 1 / np.cos((90 - np.array(angles.ele_val)) / 180 * np.pi)
    plt.errorbar(rel_thickness, T_sky+273.15, yerr=T_sky_err, color=c, fmt='.-', markeredgecolor ="black",label='{}GHz'.format(freq+4.3) ,ecolor='black', capthick=2,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
plt.xlabel("rel. thickness [-]")
plt.ylabel("T [K°]")
plt.legend()
plt.tight_layout()
plt.savefig("sky_temp_thickness.pdf")
plt.show()