import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def linear(x, m, b):
    return x*m + b


df01 = pd.read_csv("0_1.txt", names=["v", "f", "err"])
df1 = pd.read_csv("1.txt", names=["v", "f", "err"])

fig, ax = plt.subplots(1)
ax.errorbar(np.abs(df1.v), df1.f, yerr=df1.err, color="red", label="dt = 1 ms", fmt='o', markeredgecolor="black", ecolor='black', capthick=2, capsize=2, elinewidth=1, markeredgewidth=0.5, ms=5)
ax.errorbar(np.abs(df01.v), df01.f, yerr=df01.err, color="black", label="dt = 0.1 ms", fmt='o', markeredgecolor="black", ecolor='black', capthick=2, capsize=2, elinewidth=1, markeredgewidth=0.5, ms=5)


# drop all values that don't make sense, drop all values where frequency = 0
# because there std is also = 0, thus making the weight infinite
v1 = np.abs(df1.v)
mask = (v1 < 3) & (v1 > 0) & (df1.f != 0)
v1 = v1[mask]
f1 = np.array(df1.f)[mask]
err1 = np.array(df1.err)[mask]

v2 = np.abs(df01.v)
mask = (v2 > 0.3) & (df01.f != 0)
v2 = v2[mask]
f2 = np.array(df01.f)[mask]
err2 = np.array(df01.err)[mask]


popt1, pcov1 = curve_fit(linear, v1, f1, sigma=err1)
ax.plot(np.abs(df1.v), linear(np.abs(df1.v), *popt1), color="red", ls="--")

popt01, pcov01 = curve_fit(linear, v2, f2, sigma=err2)
ax.plot(np.abs(df01.v), linear(np.abs(df01.v), *popt01), color="black", ls="--")

print("\ndt = 1 ms on oscilloscope:")
print(f"slope = ({popt1[0]:.2f} +/- {np.sqrt(pcov1[0][0]):.2f}) krad/deg")
print(f"offset = ({popt1[1]:.2f} +/- {np.sqrt(pcov1[1][1]):.2f}) krad/s")

print("\ndt = 0.1 ms on oscilloscope:")
print(f"slope = ({popt01[0]:.2f} +/- {np.sqrt(pcov01[0][0]):.2f}) krad/deg")
print(f"offset = ({popt01[1]:.2f} +/- {np.sqrt(pcov01[1][1]):.2f}) krad/s")

ax.set_xlabel(r"$v$ [deg/s]")
ax.set_ylabel(r"$f$ [krad/s]")
plt.legend()
plt.savefig("slope.pdf")
plt.show()