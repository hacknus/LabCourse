import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    t = np.linspace(0,2*1e-9,1000)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

    omega_1 = 22.3e9
    omega_2 = 18.55e9

    RF = np.cos(omega_1*t)

    ax0.plot(t, RF, label=f"{omega_1*1e-9} GHz")
    ax0.plot(t, np.cos(omega_2*t), label=f"{omega_2*1e-9} GHz")
    ax0.set_ylabel(r"$A$ [-]")
    ax0.legend()

    ax1.plot(t, np.cos(omega_2*t)*RF, color="orange", ls="--",label="product")
    #ax1.plot(t, np.cos((omega_1 - omega_2)*t)/2+np.cos((omega_1 + omega_2)*t)/2, color="orange", ls="--",label="product")
    ax1.plot(t, np.cos((omega_1 - omega_2)*t)/2, color="red", label="difference")
    ax1.plot(t, np.cos((omega_1 + omega_2)*t)/2, color="black", label="sum")
    ax1.legend()
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"$A$ [-]")
    plt.tight_layout()
    plt.savefig("mixing.pdf")
    plt.show()
