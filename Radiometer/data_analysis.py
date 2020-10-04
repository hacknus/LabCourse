# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform

OS = platform.system()

if OS == "Windows":
    angles = pd.read_csv(r'16GHZ\angles.csv')
    hot_load_before = pd.read_csv(r'16GHZ\hot_load_before.csv')
    hot_load_after = pd.read_csv(r'16GHZ\hot_load_after.csv')
else:
    angles = pd.read_csv('16GHZ/angles.csv')
    hot_load_before = pd.read_csv('16GHZ/hot_load_before.csv')
    hot_load_after = pd.read_csv('16GHZ/hot_load_after.csv')

k=0                     #wie viele datenpunkte weglassen
T0=2.7 + 273.15
tau0=0.3
T_m = 16-10

U_cold = angles.v_val[9]
T_hot = (hot_load_before.T_load_val+hot_load_after.T_load_val)/2.
U_hot = (hot_load_before.v_val+hot_load_after.v_val)/2.

def tau_function(tau0):
    T_cold =  T0*np.exp(-tau0) + (1-np.exp(-tau0))*T_m  
    tau = np.zeros(10-k)
    for i in np.arange(10-k):
        U_teta = angles.v_val[i+k]
        T_teta = (U_teta-U_hot)/(U_hot-U_cold) * (T_hot-T_cold)+T_hot
        tau[i] = np.log((T_m-T0)/(T_m-T_teta))
    return tau

rel_thickness = np.zeros(10-k)
for i in np.arange(10-k):
    rel_thickness[i] = 1/np.cos((90-angles.ele_val[i+k])/180*np.pi)

B0=1
while np.abs(B0) >= 0.01:
    tau = tau_function(tau0)
    sum1=0
    sum2=0
    for i in np.arange(10-k):
        sum1 = sum1 + (rel_thickness[i]-np.mean(rel_thickness))*(tau[i]-np.mean(tau))
        sum2 = sum2 + (rel_thickness[i]-np.mean(rel_thickness))**2
    B1 = sum1/sum2
    B0 = np.mean(tau) - B1* np.mean(rel_thickness)    
    tau0 = B0 + B1*1
    
    f= np.zeros(4)
    for i in np.arange(4):
        f[i] = B0 + B1*i

plt.plot(f)
plt.plot(rel_thickness,tau)
plt.show()
print(tau0)
