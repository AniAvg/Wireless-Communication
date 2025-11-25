import numpy as np
import matplotlib.pyplot as plt

symbols = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
s = symbols[0]

f = 2500
t = np.linspace(0, 5e-3, 2000)
A_i = np.real(s)
A_q = np.imag(s)

I = A_i * np.cos(2 * np.pi * f * t)
Q = A_q * np.sin(2 * np.pi * f * t)
s_t = I - Q

snr_db = 5  # db = decibels
snr_linear = 10**(snr_db / 10)
signal_power = np.mean(np.abs(symbols)**2)
noise_power = signal_power / snr_linear

noise_std = np.sqrt(noise_power / 2)

num_symbols = 1000
random_symbols = np.random.choice(symbols, size=num_symbols)

noise = np.random.normal(0, np.sqrt(noise_power), size=s_t.shape)
s_t_noisy = s_t + noise

noise = np.random.normal(0, noise_std, size=num_symbols) + \
       1j * np.random.normal(0, noise_std, size=num_symbols)

noisy_symbols = random_symbols + noise


# Plotting I, Q, Combined signal
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, I, color="red", label = "I")
plt.plot(t, Q, label = "Q")
plt.title("In phase and Quadrature carrier")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, s_t)
plt.title("4QAM signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, s_t_noisy)
plt.title("4QAM with noise")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting Constellation diagram
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(np.real(symbols), np.imag(symbols), s=100)
plt.axhline(0, color='gray', linewidth=0.7)
plt.axvline(0, color='gray', linewidth=0.7)
plt.grid(True)
plt.title("4QAM Constellation")
plt.xlabel("In phase (I)")
plt.ylabel("Quadrature (Q)")

plt.subplot(1, 2, 2)
plt.scatter(np.real(noisy_symbols), np.imag(noisy_symbols), s=100, alpha = 0.7)
plt.axhline(0, color='gray', linewidth=0.7)
plt.axvline(0, color='gray', linewidth=0.7)
plt.grid(True)
plt.title("4QAM Constellation with noise")
plt.xlabel("In phase (I)")
plt.ylabel("Quadrature (Q)")

plt.tight_layout()
plt.show()
