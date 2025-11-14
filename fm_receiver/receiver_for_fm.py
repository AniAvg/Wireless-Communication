import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import hilbert

msg_signal, sample_rate = sf.read("original_audio.mp3")

if msg_signal.ndim > 1:
    msg_signal = msg_signal[0]

####
n = np.max(np.abs(msg_signal))
if n > 0:
    msg_signal = msg_signal / n
else:
    raise ValueError("Audio signal is zero")
#####

msg_signal = msg_signal / np.max(np.abs(msg_signal))
T = 1 / sample_rate
freq = np.fft.fftfreq(len(msg_signal), T)
Am = np.max(np.abs(msg_signal))

carrier_freq = 5 * np.max(np.abs(freq))
Ac = 1
time = np.arange(len(msg_signal)) / sample_rate

kf = 100
integral_m_t = np.cumsum(msg_signal) * (1 / sample_rate)

fm_signal = Ac * np.cos(2 * np.pi * carrier_freq * time + 2 * np.pi * kf * integral_m_t)

analytic = hilbert(fm_signal)
phase = np.unwrap(np.angle(analytic))
inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
recovered = inst_freq - carrier_freq
recovered = recovered / kf

sf.write("recovered_audio.mp3", recovered, sample_rate)

def adding_awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power =signal_power / snr_linear
    std_dev = np.sqrt(noise_power)
    noise = std_dev * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

noisy_fm_signal = adding_awgn(fm_signal, 30)

analytic = hilbert(noisy_fm_signal)
phase = np.unwrap(np.angle(analytic))
inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
recovered_noisy = inst_freq - carrier_freq
recovered_noisy = recovered_noisy / kf

sf.write("recovered_noisy_audio.mp3", recovered_noisy, sample_rate)

start_time = 0
end_time = 0.01

start = int(start_time * sample_rate)
end = int(end_time * sample_rate)

def plotting(t, signal, title):
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)


plt.figure(figsize = (12, 8))
plt.subplot(3, 1, 1)
plotting(time, msg_signal, "Message Signal")
plt.subplot(3, 1, 2)
plotting(time, fm_signal, "FM Signal")
plt.subplot(3, 1, 3)
plotting(time, recovered, "Recovered Signal")

plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(time[start:end], msg_signal[start:end], "Message Signal (Zoomed)")
plt.subplot(2, 1, 2)
plotting(time[start:end], recovered[start:end], "Recovered Signal (Zoomed)")

plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(time, noisy_fm_signal, "Noisy Signal")
plt.subplot(2, 1, 2)
plotting(time[start:end], noisy_fm_signal[start:end], "Noisy Signal (Zoomed)")

 # Recovered Signal
plt.figure(figsize = (18, 8))
plt.subplot(2, 2, 1)
plotting(time, recovered, "Recovered Message Signal")
plt.subplot(2, 2, 2)
plotting(time[start:end], recovered[start:end], "Recovered Message Signal (Zoomed)")
plt.subplot(2, 2, 3)
plotting(time, recovered_noisy, "Recovered Noisy Signal")
plt.subplot(2, 2, 4)
plotting(time[start:end], recovered_noisy[start:end], "Recovered Noisy Signal (Zoomed)")

plt.tight_layout()
plt.show()