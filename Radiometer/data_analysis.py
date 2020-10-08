# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
from scipy.optimize import curve_fit
import csv

def T_cold(T0, tau_i, T_m):
    return T0 * np.exp(-tau_i) + (1 - np.exp(-tau_i)) * T_m

def T_cold_err(T0, tau_i, tau_i_err, T_m):
    return np.sqrt((np.exp(-tau_i) * (T0-T_m) * tau_i_err) ** 2)

def tau(U_theta, U_cold, U_hot, T_hot, tau_i):
    T0 = 2.7
    T_m = 16 - 10 + 273.15
    T_theta = (U_theta - U_hot) / (U_hot - U_cold) * (T_hot - T_cold(T0, tau_i, T_m)) + T_hot
    return np.log((T_m - T0) / (T_m - T_theta))

def coefficient(U_hot,U_cold,T_hot,T_cold):
    return (T_hot-T_cold)/(U_hot-U_cold)

def coefficient_err(U_hot, dU_hot, U_cold, dU_cold, T_hot, dT_hot, T_cold, dT_cold):
    return np.sqrt( (dT_hot / (U_hot-U_cold)) ** 2 + ( dT_cold /(U_hot-U_cold)) ** 2 + ((T_hot-T_cold)/(U_hot-U_cold)**2*dU_hot**2) + ((T_hot-T_cold)/(U_hot-U_cold)**2*dU_cold**2))

def T_offset_err(U_hot, dU_hot, dT_hot, K, Kerr):
    return np.sqrt( dT_hot**2 + (U_hot * Kerr)**2 + (K * dU_hot)**2)

def T_offset(U_hot,T_hot,K):
    return T_hot - K*U_hot

def linear(t, m, b):
    return t * m + b


OS = platform.system()

for freq, c in zip([16, 17, 18, 19], ['red', 'blue', 'green', 'orange']):

    print("calculating frequency = {} GHZ".format(freq))
    if OS == "Windows":
        angles = pd.read_csv(r'{}GHZ\angles.csv'.format(freq))
        hand = pd.read_csv(r'{}GHZ\hand.csv'.format(freq))
        hot_load_before = pd.read_csv(r'{}GHZ\hot_load_before.csv'.format(freq))
        hot_load_after = pd.read_csv(r'{}GHZ\hot_load_after.csv'.format(freq))
    else:
        angles = pd.read_csv('{}GHZ/angles.csv'.format(freq))
        hand = pd.read_csv('{}GHZ/hand.csv'.format(freq))
        hot_load_before = pd.read_csv('{}GHZ/hot_load_before.csv'.format(freq))
        hot_load_after = pd.read_csv('{}GHZ/hot_load_after.csv'.format(freq))

    U_cold = np.array(angles.v_val)[-1]
    dU_cold = np.array(angles.v_std)[-1]
    T_hot = np.array(hot_load_before.T_load_val + hot_load_after.T_load_val)[0] / 2. + 273.15
    dT_hot = np.sqrt(np.array(hot_load_before.T_load_std**2 + hot_load_after.T_load_std**2)[0])
    U_hot = np.array(hot_load_before.v_val + hot_load_after.v_val)[0] / 2.
    dU_hot = np.sqrt(np.array(hot_load_before.v_std**2 + hot_load_after.v_std**2)[0])
    rel_thickness = 1 / np.cos((90 - np.array(angles.ele_val)) / 180 * np.pi)

    tau_i = 0.3

    # perform first iteration
    taus = tau(np.array(angles.v_val), U_cold, U_hot, T_hot, tau_i)
    popt, pcov = curve_fit(linear, rel_thickness, taus, p0=[tau_i, 0])

    while abs(popt[1]) > 1e-2:
        taus = tau(np.array(angles.v_val), U_cold, U_hot, T_hot, tau_i)
        # TODO: error propagationneeds to be implemented

        popt, pcov = curve_fit(linear, rel_thickness, taus, p0=[tau_i, 0])
        tau_i = popt[0]
        tau_err = pcov[0][0]
        offset = popt[1]

    T0 = 2.7
    T_m = 16 - 10 + 273.15

    T_hand = (np.array(hand.v_val)[0] - U_hot) / (U_hot - U_cold) * (T_hot - T_cold(T0, tau_i, T_m)) + T_hot
    T_hand_err = 5


    plt.plot(rel_thickness, 100*taus, label="{} GHZ".format(freq+3.75+0.55), color=c)
    rel_thickness = np.linspace(0, 4, 10)
    plt.plot(rel_thickness, 100*linear(rel_thickness, *popt), color=c, ls="--")
    print("tau = {:.6f} +/- {:.6f}".format(tau_i, tau_err))
    print("T_hand = {:.6f} +/- {:.6f}".format(T_hand - 273.15, T_hand_err))
    K = coefficient(U_hot,U_cold,T_hot,T_cold(T0, tau_i, T_m))
    dT_cold = T_cold_err(T0, tau_i, tau_err, T_m)
    K_err = coefficient_err(U_hot, dU_hot, U_cold, dU_cold, T_hot, dT_hot, T_cold(T0, tau_i, T_m), dT_cold)
    C = T_offset(U_hot,T_hot,K)
    C_err = T_offset_err(U_hot, dU_hot, dT_hot, K, K_err)
    if OS == "Windows":
        with open('{}GHZ\K_and_C.csv'.format(freq), 'w', newline='') as csvfile:
            fieldnames = ['K', 'K_err','C', 'C_err']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'K':K , 'K_err': K_err, 'C': C, 'C_err': C_err})
    else:
        with open('{}GHZ/K_and_C.csv'.format(freq), 'w', newline='') as csvfile:
            fieldnames = ['K', 'K_err','C', 'C_err']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'K':K , 'K_err': K_err, 'C': C, 'C_err': C_err})
    
plt.xlabel("rel. thickness [-]")
plt.ylabel("opacity [-]")
plt.legend()
plt.show()
