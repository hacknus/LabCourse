import numpy as np
import matplotlib.pyplot as plt

def get_ft(sig,sF):
    fT = np.fft.fft(sig) / len(sig)
    fT = np.abs(fT[range(int(len(sig) / 2))])
    frequencies = np.arange(len(sig) / 2) / len(sig) * sF
    return frequencies,fT

def gauss(sigma,mu,x):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))

omega = np.linspace(0,50,1000)
amplitude = gauss(1,22,omega) + 30*gauss(40,80,omega) + 0.2*gauss(0.1,2.4,omega)
amplitude_no_wifi = gauss(1,22,omega) + 30*gauss(40,80,omega)

omega2 = 18.55
samplingFreq = 1000
t = np.arange(0, 10, 1/samplingFreq)

RF = np.sum(amplitude[:,None]*np.cos(2*np.pi*omega[:,None] * t),axis=0)

LO = np.cos(2*np.pi*omega2 * t)

frequencies, fT = get_ft(RF, samplingFreq)
frequencies2, LOfT = get_ft(LO, samplingFreq)
frequencies3, IF = get_ft(LO*RF, samplingFreq)

fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.plot(t,RF/len(amplitude)*10,label="RF")
#ax1.plot(t[::2],np.sum(IF*np.cos(2*np.pi*frequencies3 * t[:,None]),axis=0))
ax1.set_xlim(0,2)
ax1.set_ylabel("Amplitude [-]")
ax1.set_xlabel(r"t [$10^{-9}$s]")
ax1.legend(loc="upper right")

ax2.plot(frequencies,fT,label="FT RF")
#ax2.plot(frequencies2,LOfT)
ax2.plot(frequencies3,IF,ls="--",label="FT IF")
ax2.plot([3.75,3.75],[0,0.2],color="red",ls=":")
ax2.plot([omega2,omega2],[0,0.2],color="green",ls=":")
ax2.set_xlim(0,50)
ax2.set_ylabel("Fourier Space")
ax2.set_xlabel(r"$f$ [GHz]")
ax2.text(2.4-1,0.5,"WIFI")
ax2.text(3.75-0.5,0.3,"BP",color="red")
ax2.text(18.55-1,0.3,"LO",color="green")
ax2.text(22-1,0.5,r"H$_2$O")
ax2.text(47-1,0.5,r"O$_2$")
ax2.set_ylim(0,0.6)
ax2.legend(loc="lower right")

frequencies3, IF = get_ft(LO*RF, samplingFreq)
IF = IF[(frequencies3 < 3.75 + 0.1) & (frequencies3 > 3.75-0.1)]
frequencies3 = frequencies3[(frequencies3 < 3.75 + 0.1) & (frequencies3 > 3.75-0.1)]
print(np.sum(IF[:,None]*np.cos(2*np.pi*frequencies3[:,None] * t),axis=0))
ax3.plot(t,np.sum(IF[:,None]*np.cos(2*np.pi*frequencies3[:,None] * t),axis=0)*5,label="IF")
ax3.set_xlim(0,2)
ax3.set_ylabel("Amplitude [-]")
ax3.set_xlabel(r"t [$10^{-9}$s]")
ax3.legend(loc="upper right")

plt.tight_layout()
plt.savefig("mixing2.pdf")
plt.show()
