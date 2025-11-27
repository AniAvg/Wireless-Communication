import numpy as np
import matplotlib.pyplot as plt

fs = 50000
T = 0.02

time = np.arange(0, 0.02, 1/fs)

# Message signal
fm = 200
Am = 1
m_t = Am * np.cos(2 * np.pi * fm * time)

fft_values_m = np.fft.fft(m_t) / len(m_t)
freq_m = np.fft.fftfreq(len(m_t), 1/fs)
magnitude_m = np.abs(fft_values_m) * 2

# Carrier signal
fc = 2000
Ac = 2

c_t = Ac * np.cos(2 * np.pi * fc * time)
fft_values_c = np.fft.fft(c_t)
freq_c = np. fft.fftfreq(len(c_t), 1/fs)
magnitude_c = np.abs(fft_values_c)

# Amplitude Modulated signal
s_t = (Ac + Am * np.cos(2 * np.pi * fm * time)) * np.cos(2 * np.pi * fc * time)
fft_values_s = np.fft.fft(s_t)
freq_s = np.fft.fftfreq(len(s_t), 1/fs)
magnitude_s = np.abs(fft_values_s)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(freq_m, magnitude_m)
plt.title("Message Signal")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(freq_c, magnitude_c)
plt.title("Carrier signal")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(freq_s, magnitude_s)
plt.title("Amplitude Modulated Signal")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
