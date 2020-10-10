import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    t = np.linspace(0,1*np.pi,1000)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

    omega_1 = 22.3
    omega_2 = 18.55

    ax0.plot(t, np.cos(omega_1*t), label=f"{omega_1} GHz")
    ax0.plot(t, np.cos(omega_2*t), label=f"{omega_2} GHz")
    ax0.set_ylabel(r"$A$ [-]")
    ax0.legend()

    ax1.plot(t, np.cos(omega_2*t)*np.cos(omega_1*t), color="red", ls="--",label="product")
    ax1.plot(t, np.cos((omega_1-omega_2)*t), color="red", label="difference")
    ax1.plot(t, np.cos((omega_1+omega_2)*t), color="black", label="sum")
    ax1.legend()
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"$A$ [-]")
    plt.tight_layout()
    plt.savefig("mixing.pdf")
    plt.show()
