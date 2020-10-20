import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from math import log10, floor

def tau_mat(T_m, T_in, T_out):
    return np.log((T_m - T_in) / (T_m - T_out))

def tau_mat_err(T_m,T_in,T_out,T_m_err,T_in_err,T_out_err):
    k = (T_m-T_out)/(T_m-T_in)
    a = k*(1/(T_m-T_out)-(T_m-T_in)/(T_m-T_out)**2)*T_m_err
    b = k* 1/(T_m-T_out)*T_in_err
    c = k*(T_m-T_in)/(T_m-T_out)**2*T_out_err
    return np.sqrt(a**2+b**2+c**2)

def trans(tau):
    return np.e**(-tau)

def trans_err(tau,tau_err):
    return np.e**(-tau)*tau_err

def open_T_in(file):
    T=np.zeros(4)
    T[0]=file.T_20[9]
    T[1]=file.T_21[9]
    T[2]=file.T_22[9]
    T[3]=file.T_23[9]
    return T

def open_T_in_err(file):
    T=np.zeros(4)
    T[0]=file.T_20_err[9]
    T[1]=file.T_21_err[9]
    T[2]=file.T_22_err[9]
    T[3]=file.T_23_err[9]
    return T

def open_T_out(file):
    T=np.zeros((6,4))
    for i in np.arange(6):
        T[i][0]= file.T16[i]
        T[i][1]= file.T17[i]
        T[i][2]= file.T18[i]
        T[i][3]= file.T19[i]
    return T
      
def open_T_out_err(file):
    T=np.zeros((6,4))
    for i in np.arange(6):
        T[i][0]= file.T16err[i]
        T[i][1]= file.T17err[i]
        T[i][2]= file.T18err[i]
        T[i][3]= file.T19err[i]
    return T


def sign(x): 
    if x >= 1:
        return 2
    else: 
        return -(int(floor(log10(abs(x)))))+1
def emis(T_out,T_in,T_m,t):
    return ((T_out+273.15)-t*(T_in+273.15))/(T_m+273.15)

def emis_err(T_out,T_out_err,t,t_err,T_in,T_in_err,T_m,T_m_err):
    T_out=T_out+273.15
    T_in=T_in+273.15
    T_m = T_m+273.15
    k1 = 1/T_m*T_out_err
    k2 = T_in/T_m*t_err
    k3 = t/T_m*T_in_err
    k4 = (T_out-t*T_in)/T_m**2*T_m_err
    return np.sqrt(k1**2+k2**2+k3**2+k4**2)
  

tau = np.zeros((6,4))
tau_err = np.zeros((6,4))
t = np.zeros((6,4))
t_err = np.zeros((6,4))
e = np.zeros((6,4))
e_err = np.zeros((6,4))


T_in = open_T_in(pd.read_csv(r'sky_temp.csv'))
T_in_err = open_T_in_err(pd.read_csv(r'sky_temp.csv'))
T_out = open_T_out(pd.read_csv(r'materials.csv'))
T_out_err = open_T_out_err(pd.read_csv(r'materials.csv'))
T_mat = np.array([36,36,16,16,16,16])
T_mat_err = np.array([1,1,1,1,1,1])

material = ["hand", "2 hands", "blackbody", "blue foam", "cellphone", "acrylic"]

for k in np.arange(6):
    for j in np.arange(4):
        tau[k][j] = tau_mat(T_mat[k],T_in[j],T_out[k][j])

for k in np.arange(6):
    for j in np.arange(4):
        tau_err[k][j] = tau_mat_err(T_mat[k],T_in[j],T_out[k][j],T_mat_err[k],T_in_err[j],T_out_err[k][j])

for k in np.arange(6):
    for j in np.arange(4):
        t[k][j] = trans(tau[k][j])

for k in np.arange(6):
    for j in np.arange(4):
        t_err[k][j] = trans_err(tau[k][j],tau_err[k][j])

e = np.ones((6,4))-t
e_err=t_err

print(e)
print(e_err)

fig, (ax0, ax1) = plt.subplots(ncols=2)
for directory, freq, c in zip(["16GHZ", "17GHZ", "18GHZ", "19GHZ"], np.arange(16.55+3.75, 23.75, 1), ['red', 'blue', 'green', 'orange']):
    
    ax0.errorbar(0+freq/10, tau[0][int(freq-20.3)], yerr=tau_err[0][int(freq-20.3)], color=c, label=f"{freq} GHz", fmt='o', markeredgecolor ="black",ecolor='black', capthick=2,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax0.errorbar(1+freq/10, tau[1][int(freq-20.3)], yerr=tau_err[1][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax0.errorbar(2+freq/10, tau[2][int(freq-20.3)], yerr=tau_err[2][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(0+freq/10, tau[3][int(freq-20.3)], yerr=tau_err[3][int(freq-20.3)], color=c, label=f"{freq} GHz", fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(1+freq/10, tau[4][int(freq-20.3)], yerr=tau_err[4][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(2+freq/10, tau[5][int(freq-20.3)], yerr=tau_err[5][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
ax0.set_xticks(np.arange(3)+2.2)
ax0.set_xticklabels(["hand", "2 hands", "blackbody"])
ax1.set_xticks(np.arange(3)+2.2)
ax1.set_xticklabels(["blue foam", "cellphone", "acrylic"])
ax0.set_ylabel(r'$\tau$ [-]')
ax1.set_ylabel(r"$\tau$ [-]")
ax0.set_xlabel("material")
ax1.set_xlabel("material")
plt.legend()
plt.tight_layout()
plt.savefig("material_tau.pdf")
plt.show()

fig, (ax0, ax1) = plt.subplots(ncols=2)
for directory, freq, c in zip(["16GHZ", "17GHZ", "18GHZ", "19GHZ"], np.arange(16.55+3.75, 23.75, 1), ['red', 'blue', 'green', 'orange']):
    
    ax0.errorbar(0+freq/10, t[0][int(freq-20.3)], yerr=t_err[0][int(freq-20.3)], color=c, label=f"{freq} GHz", fmt='o', markeredgecolor ="black",ecolor='black', capthick=2,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax0.errorbar(1+freq/10, t[1][int(freq-20.3)], yerr=t_err[1][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax0.errorbar(2+freq/10, t[2][int(freq-20.3)], yerr=t_err[2][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(0+freq/10, t[3][int(freq-20.3)], yerr=t_err[3][int(freq-20.3)], color=c, label=f"{freq} GHz", fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(1+freq/10, t[4][int(freq-20.3)], yerr=t_err[4][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(2+freq/10, t[5][int(freq-20.3)], yerr=t_err[5][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
ax0.set_xticks(np.arange(3)+2.2)
ax0.set_xticklabels(["hand", "2 hands", "blackbody"])
ax1.set_xticks(np.arange(3)+2.2)
ax1.set_xticklabels(["blue foam", "cellphone", "acrylic"])
ax0.set_ylabel(r"$t$ [-]")
ax1.set_ylabel(r"$t$ [-]")
ax0.set_xlabel("material")
ax1.set_xlabel("material")
plt.legend()
plt.tight_layout()
plt.savefig("material_transmittance.pdf")
plt.show()

fig, (ax0, ax1) = plt.subplots(ncols=2)

for directory, freq, c in zip(["16GHZ", "17GHZ", "18GHZ", "19GHZ"], np.arange(16.55+3.75, 23.75, 1), ['red', 'blue', 'green', 'orange']):
    
    ax0.errorbar(0+freq/10, e[0][int(freq-20.3)], yerr=e_err[0][int(freq-20.3)], color=c, label=f"{freq} GHz", fmt='o', markeredgecolor ="black",ecolor='black', capthick=2,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax0.errorbar(1+freq/10, e[1][int(freq-20.3)], yerr=e_err[1][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax0.errorbar(2+freq/10, e[2][int(freq-20.3)], yerr=e_err[2][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(0+freq/10, e[3][int(freq-20.3)], yerr=e_err[3][int(freq-20.3)], color=c, label=f"{freq} GHz", fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(1+freq/10, e[4][int(freq-20.3)], yerr=e_err[4][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
    ax1.errorbar(2+freq/10, e[5][int(freq-20.3)], yerr=e_err[5][int(freq-20.3)], color=c, fmt='o', markeredgecolor ="black",ecolor='black', capthick=2 ,capsize=2, elinewidth=1, markeredgewidth=0.5,   ms=5)
ax0.set_xticks(np.arange(3)+2.2)
ax0.set_xticklabels(["hand", "2 hands", "blackbody"])
ax1.set_xticks(np.arange(3)+2.2)
ax1.set_xticklabels(["blue foam", "cellphone", "acrylic"])
ax0.set_ylabel(r"$\epsilon$ [-]")
ax1.set_ylabel(r"$\epsilon$ [-]")
ax0.set_xlabel("material")
ax1.set_xlabel("material")
plt.legend()
plt.tight_layout()
plt.savefig("material_emissivity.pdf")
plt.show()


for k in np.arange(6):
    for j in np.arange(4):
        tau[k][j] = round(tau[k][j],sign(tau_err[k][j]))
        tau_err[k][j] = round(tau_err[k][j],sign(tau_err[k][j]))
        t[k][j] = round(t[k][j],sign(t_err[k][j]))
        t_err[k][j] = round(t_err[k][j],sign(t_err[k][j]))
        e[k][j] = round(e[k][j],sign(e_err[k][j]))
        e_err[k][j] = round(e_err[k][j],sign(e_err[k][j]))

with open('material_tau.csv','w', newline='') as csvfile:
    fieldnames = ['Material','tau16','tau16err','tau17','tau17err','tau18','tau18err','tau19','tau19err']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in np.arange(6):
        writer.writerow({'Material':material[i], 'tau16':tau[i][0],'tau16err':tau_err[i][0],'tau17':tau[i][1],'tau17err':tau_err[i][1],'tau18':tau[i][2],'tau18err':tau_err[i][2],'tau19':tau[i][3],'tau19err':tau_err[i][3]})

with open('material_t.csv','w', newline='') as csvfile:
    fieldnames = ['Material','t16','t16err','t17','t17err','t18','t18err','t19','t19err']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in np.arange(6):
        writer.writerow({'Material':material[i], 't16':t[i][0],'t16err':t_err[i][0],'t17':t[i][1],'t17err':t_err[i][1],'t18':t[i][2],'t18err':t_err[i][2],'t19':t[i][3],'t19err':t_err[i][3]})
        
with open('material_e.csv','w', newline='') as csvfile:
    fieldnames = ['Material','e16','e16err','e17','e17err','e18','e18err','e19','e19err']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in np.arange(6):
        writer.writerow({'Material':material[i], 'e16':e[i][0],'e16err':e_err[i][0],'e17':e[i][1],'e17err':e_err[i][1],'e18':e[i][2],'e18err':e_err[i][2],'e19':e[i][3],'e19err':e_err[i][3]})        
        

