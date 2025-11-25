import numpy as np
import matplotlib.pyplot as plt

fs = 50000
T = 0.02

time = np.arange(0, 0.02, 1/fs)

# Message signal
fm = 200
Am = 1
m_t = Am * np.cos(2 * np.pi * fm * time)
# Carrier signal
fc = 1000
Ac = 2
c_t = Ac * np.cos(2 * np.pi * fc * time)
# Amplitude Modulated signal
s_t = (Ac + Am * np.cos(2 * np.pi * fm * time)) * np.cos(2 * np.pi * fc * time)

plt.figure(figsize=(10,8))

plt.subplot(3, 1, 1)
plt.plot(time, m_t)
plt.title("Message Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, c_t)
plt.title("Carrier signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, s_t)
plt.title("Amplitude Modulated Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()