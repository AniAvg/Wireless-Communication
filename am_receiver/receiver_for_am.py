import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import hilbert

data, sample_rate = sf.read("original_audio.mp3")

# checking if the audio is mono
if data.ndim > 1:
    data = data[:, 0]

data = data / np.max(np.abs(data))
# T = time step
T = 1 / sample_rate
freq = np.fft.fftfreq(len(data), T)
Am = np.max(np.abs(data))

carrier_freq = 2000
Ac = 2
time = np.arange(len(data)) / sample_rate

carrier_signal = Ac * np.cos(2 * np.pi * carrier_freq * time)
# s(t) = (Ac + k * m(t)) * cos(2 * pi * fc * t)
mod_index = Am / Ac
am_signal = (1 + mod_index * data) * carrier_signal



analytic_signal = hilbert(am_signal)
envelope = np.abs(analytic_signal)
recovered = (envelope / Ac - 1) / mod_index
recovered = np.clip(recovered, -1, 1)

sf.write("recovered_audio.mp3", recovered, sample_rate)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(time, data)
plt.title("Message signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(time, am_signal)
plt.title("AM Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(time, recovered)
plt.title("Recovered Message Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()