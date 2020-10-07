# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from scipy.optimize import curve_fit



def T_cold(tau_i):
    T0 = 2.7 - 273.15
    T_m = 16 - 10
    return T0 * np.exp(-tau_i) + (1 - np.exp(-tau_i)) * T_m
    
def tau(U_theta, U_cold, U_hot, T_hot, tau_i):
    T0 = 2.7 - 273.15
    T_m = 16 - 10
    T_theta = (U_theta - U_hot) / (U_hot - U_cold) * (T_hot - T_cold(tau_i)) + T_hot
    return np.log((T_m - T0) / (T_m - T_theta))

def coefficient(U_hot,U_cold,T_hot,T_cold):
    return (T_hot-T_cold)/(U_hot-U_cold)

def T_offset(U_hot,T_hot,K):
    return T_hot - K*U_hot

def linear(t, m, b):
    return t * m + b


OS = platform.system()

for freq, c in zip([16, 17, 18, 19], ['red', 'blue', 'green', 'orange']):

    print("calculating frequency = {} GHZ".format(freq))
    if OS == "Windows":
        angles = pd.read_csv(r'{}GHZ\angles.csv'.format(freq))
        hot_load_before = pd.read_csv(r'{}GHZ\hot_load_before.csv'.format(freq))
        hot_load_after = pd.read_csv(r'{}GHZ\hot_load_after.csv'.format(freq))
    else:
        angles = pd.read_csv('{}GHZ/angles.csv'.format(freq))
        hot_load_before = pd.read_csv('{}GHZ/hot_load_before.csv'.format(freq))
        hot_load_after = pd.read_csv('{}GHZ/hot_load_after.csv'.format(freq))

    U_cold = np.array(angles.v_val)[-1]
    T_hot = np.array(hot_load_before.T_load_val + hot_load_after.T_load_val)[0] / 2.
    U_hot = np.array(hot_load_before.v_val + hot_load_after.v_val)[0] / 2.
    rel_thickness = 1 / np.cos((90 - np.array(angles.ele_val)) / 180 * np.pi)

    tau_i = 0.3
    last_tau = 0

    # perform first iteration
    taus = tau(np.array(angles.v_val), U_cold, U_hot, T_hot, tau_i)
    tau_err = tau(np.array(angles.v_std), U_cold, U_hot, T_hot, tau_i)
    last_tau = tau_i
    popt, pcov = curve_fit(linear, rel_thickness, taus, p0=[tau_i, 0], sigma=tau_err)


    while abs(linear(0, *popt)) > 1e-2:
        taus = tau(np.array(angles.v_val), U_cold, U_hot, T_hot, tau_i)

        # TODO: error propagationneeds to be implemented
        tau_err = tau(np.array(angles.v_std), U_cold, U_hot, T_hot, tau_i)
        last_tau = tau_i
        popt, pcov = curve_fit(linear, rel_thickness, taus, p0=[tau_i, 0], sigma=tau_err)
        tau_i = popt[0]
        tau_err = pcov[0][0]
        offset = popt[1]
    T_c = T_cold(tau_i)
    T_c_err = 5
    K = coefficient(U_hot,U_cold,T_hot,T_c)
    print(K)
    K_err=5
    C = T_offset(U_hot,T_hot,K)
    C_err=5
    plt.plot(rel_thickness, taus, label="{} GHZ".format(freq), color=c)
    rel_thickness = np.linspace(0,4,10)
    plt.plot(rel_thickness, linear(rel_thickness, *popt), color=c, ls="--")
    print("tau = {:.6f} +/- {:.6f}".format(tau_i, tau_err))
    print("T_cold = {:.6f} +/- {:.6f}".format(T_c, T_c_err))
    print("Coefficient = {:.6f} +/- {:.6f}".format(K, K_err))
    print("T_offset = {:.6f} +/- {:.6f}".format(C, C_err))
    print("")
plt.xlabel("rel thickness [-]")
plt.ylabel("opacity [%]")
plt.legend()
plt.show()

