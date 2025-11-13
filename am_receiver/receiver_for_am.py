import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

data, sample_rate = sf.read("original_audio.mp3")

# checking if the audio is mono
if data.ndim > 1:
    data = data[:, 0]

data = data / np.max(np.abs(data))
# T = time step
T = 1 / sample_rate
freq = np.fft.fftfreq(len(data), T)

carrier_freq = 40000 # carrieri frequency-n el chisht yntrel
carrier_ampl = 2
time = np.arange(len(data)) / sample_rate

carrier_signal = carrier_ampl * np.cos(2 * np.pi * carrier_freq * time)
# s(t) = (Ac + k * m(t)) * cos(2 * pi * fc * t)
# k-i arjeqy poxel (chisht yntrel)
am_signal = (1 + 0.5 * data) * carrier_signal

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time, data)
plt.title("Message signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(time, am_signal)
plt.title("AM Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()