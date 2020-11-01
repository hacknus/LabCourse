import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def linear(x, m, b):
    return x * m + b


l = 633 * 1e-9
c = 299792458.0
r = 10e-2
s = 0.4
A = s ** 2 / 4 * np.sqrt(3)
d = 0.001
L = s*3

omega = c * l ** 2 * r / (32 * np.pi * A * d)
slope = 4*A/(l*L*360)
slope_err = 1/(360*np.sqrt(3)*l)*0.01
print(f"Lock frequency: {omega:.4f}")
print(f"slope: {slope:.4f} +/- {slope_err:.4f}")

df01 = pd.read_csv("0_1.txt", names=["v", "f", "err"])
df1 = pd.read_csv("1.txt", names=["v", "f", "err"])
df1.sort_values(by=['v'], inplace=True)
df1.f[df1.v < 0] *= -1
df1.f *= 1000/(2*np.pi)
df1.err *= 1000/(2*np.pi)

fig, ax = plt.subplots(1)

# drop all values that don't make sense, drop all values where frequency = 0
# because there std is also = 0, thus making the weight infinite
v1 = df1.v

mode = -1
if mode == 0:
    mask = (v1 < 3) & (v1 > -3) & (df1.f != 0)
elif mode == 1:
    mask = (v1 < 3) & (v1 > -0) & (df1.f != 0)
elif mode == -1:
    mask = (v1 < 0) & (v1 > -3) & (df1.f != 0)

v1 = v1[mask]
f1 = np.array(df1.f)[mask]
err1 = np.array(df1.err)[mask]

popt1, pcov1 = curve_fit(linear, v1, f1, sigma=err1, absolute_sigma=True)
ax.plot(v1, linear(v1, *popt1), color="red", ls="-", label="Best fit")

ax.plot(df1.v, linear(df1.v, slope, popt1[1]), color="blue", ls="--", label="Literature")
ax.fill_between(df1.v, linear(df1.v, slope - slope_err, popt1[1]), linear(df1.v, slope + slope_err, popt1[1]),
                color="blue", alpha=0.35)

ax.axvline(omega, color='orange', label="Lock-in freq.")
ax.axvline(-omega, color='orange')

ax.errorbar(df1.v, df1.f, yerr=df1.err, color="red", label="LabView Data", fmt='o', markeredgecolor="black",
            ecolor='black', capthick=2, capsize=2, elinewidth=1, markeredgewidth=0.5, ms=3)

print("\ndt = 1 ms on oscilloscope:")
print(popt1)
print(f"slope = ({popt1[0]:.2f} +/- {np.sqrt(pcov1[0][0]):.2f}) krad/deg")
print(f"offset = ({popt1[1]:.2f} +/- {np.sqrt(pcov1[1][1]):.2f}) krad/s")

ax.set_xlabel(r"$\omega$ [°/s]")
ax.set_ylabel(r"$f$ [Hz]")
if mode == 0:
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2700, 2700)
elif mode == 1:
    ax.set_xlim(0, 2.1)
    ax.set_ylim(-100, 2500)
elif mode == -1:
    ax.set_xlim(-2.1, 0)
    ax.set_ylim(-2500, 100)
plt.legend()
if mode == 0:
    plt.savefig("Report/plots/slope.pdf")
elif mode == 1:
    plt.savefig("Report/plots/slope_pos.pdf")
elif mode == -1:
    plt.savefig("Report/plots/slope_neg.pdf")

plt.show()
