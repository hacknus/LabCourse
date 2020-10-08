import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from main import sky_temp
import csv

rel_thickness = np.array(1 / np.cos((90 - np.array(pd.read_csv(r'{}GHZ\angles.csv'.format(16)).ele_val)) / 180 * np.pi)).round(1)
for elevation, c in zip([0,3,6,9], ['red', 'blue', 'green', 'orange']):  
    T_freq = np.zeros(4)
    T_freq_err = np.zeros(4)
    frequency= np.array([16, 17, 18, 19])+3.75+0.55
    with open('temp_freq\dicke{}.csv'.format(elevation), 'w', newline='') as csvfile:
            fieldnames = ['freq', 'T [K]', 'T_err [K]']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in np.arange(4):
                angles = pd.read_csv(r'{}GHZ\angles.csv'.format(i+16))
                K_und_C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16))
                T_freq[i], T_freq_err[i]=sky_temp(angles,K_und_C,elevation)
                writer.writerow({'freq':frequency[i] , 'T [K]': T_freq[i]+273.15, 'T_err [K]': T_freq_err[i]})
    plt.errorbar(frequency, T_freq+273.15, yerr=T_freq_err, label=rel_thickness[elevation], color=c, fmt='.-', markeredgecolor ="black",ecolor='black', capthick=2,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    
plt.legend(title='thickness')
plt.ylabel('T [K°]')
plt.xticks(frequency)
plt.xlabel('frequency [GHz]')
plt.tight_layout()
plt.savefig("sky_temp_freq.pdf")
plt.show()