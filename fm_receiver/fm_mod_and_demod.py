import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt

msg_signal, sample_rate = sf.read("original_audio.mp3")

if msg_signal.ndim > 1:
    msg_signal = msg_signal[:, 0]

msg_signal = msg_signal / np.max(np.abs(msg_signal))

T = 1 / sample_rate
# freq = np.fft.fftfreq(len(msg_signal), T)

# carrier_freq = 10 * np.max(np.abs(freq))
carrier_freq = 100000
Ac = 1
time = np.arange(len(msg_signal)) / sample_rate

kf = 200

integral_msg = np.cumsum(msg_signal) * (1 / sample_rate)
fm_signal = Ac * np.cos(2 * np.pi * carrier_freq * time + 2 * np.pi * kf * integral_msg)

analytic = hilbert(fm_signal)
phase = np.unwrap(np.angle(analytic))

inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
inst_freq = np.concatenate(([inst_freq[0]], inst_freq))

recovered = (inst_freq - carrier_freq) / kf
recovered = recovered - np.mean(recovered)
recovered = recovered / np.max(np.abs(recovered))

sf.write("recovered_audio_fm.mp3", recovered, sample_rate)

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
inst_freq = np.concatenate(([inst_freq[0]], inst_freq))
recovered_noisy = (inst_freq - carrier_freq) / kf
recovered_noisy = recovered_noisy - np.mean(recovered_noisy)
recovered_noisy = recovered_noisy / np.max(np.abs(recovered_noisy))

sf.write("recovered_noisy_fm_audio.mp3", recovered_noisy, sample_rate)

start_time = 1.01
end_time = 1.02

start = int(start_time * sample_rate)
end = int(end_time * sample_rate)

def plotting(t, signal, title):
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.figure(figsize = (12, 10))
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