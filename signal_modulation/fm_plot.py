import numpy as np
import matplotlib.pyplot as plt

fs = 50000
T = 0.02

time = np.arange(0, T, 1/fs)

# Message signal
fm = 200
Am = 1
m_t = Am * np.cos(2 * np.pi * fm * time)

# Carrier signal
fc = 1000
Ac = 2
c_t = Ac * np.cos(2 * np.pi * fc * time)

# FM signal
kf = 100
integral_m_t = np.cumsum(m_t) * (1/fs)

s_t_fm = Ac * np.cos(2 * np.pi * fc * time + 2 * np.pi * kf * integral_m_t)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, m_t)
plt.title("Message Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, c_t)
plt.title("Carrier Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, s_t_fm)
plt.title("FM Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()