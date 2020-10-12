import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from main import sky_temp
import csv

with open('sky_temp.csv','w', newline='') as csvfile:
    fieldnames = ['rel. thickness', 'rel. thickness_err', 'T_20 [K]', 'T_20_err [K]', 'T_21 [K]', 'T_21_err [K]', 'T_22 [K]', 'T_22_err [K]', 'T_23 [K]', 'T_23_err [K]']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    zenith_angle = np.array((90 - np.array([17.9,19.5,21.3,23.6,26.4,30.0,34.8,41.8,53.1,90]))/180*np.pi)
    rel_thickness = np.array(1/np.cos(zenith_angle)).round(2)
    rel_thickness_err = (np.array(1/np.cos(zenith_angle))**2*np.array(np.sin(zenith_angle))*(1/180*np.pi)).round(2)
    for elevation, c in zip(np.arange(10), ['red','red','red', 'blue','blue','blue', 'green', 'green', 'green', 'orange']):  
        T_freq = np.zeros(4)
        T_freq_err = np.zeros(4)
        frequency= np.array([16, 17, 18, 19])+3.75+0.55
        for i in np.arange(4):
            angles = pd.read_csv(r'{}GHZ\angles.csv'.format(i+16))
            K_und_C = pd.read_csv(r'{}GHZ\K_and_C.csv'.format(i+16))
            T_freq[i], T_freq_err[i]=sky_temp(angles,K_und_C,elevation)
        
        if elevation in [0,3,6,9]:
            plt.errorbar(frequency, T_freq+273.15, yerr=T_freq_err, label='{0:.2f}$\pm${1:.2f}'.format(rel_thickness[elevation],rel_thickness_err[elevation]), color=c, fmt='.-', markeredgecolor ="black",ecolor='black', capthick=2,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
        T_freq = np.round(T_freq,2)
        T_freq_err = np.round(T_freq_err,2)   
        writer.writerow({'rel. thickness':rel_thickness[elevation], 'rel. thickness_err':rel_thickness_err[elevation] , 'T_20 [K]': T_freq[0], 'T_20_err [K]': T_freq_err[0], 'T_21 [K]': T_freq[1], 'T_21_err [K]': T_freq_err[1], 'T_22 [K]': T_freq[2], 'T_22_err [K]': T_freq_err[2], 'T_23 [K]': T_freq[3], 'T_23_err [K]': T_freq_err[3]})
plt.legend(title='thickness')
plt.ylabel('T [K°]')
plt.xticks(frequency)
plt.xlabel('frequency [GHz]')
plt.tight_layout()
plt.savefig("sky_temp_freq.pdf")
plt.show()