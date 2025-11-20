import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import hilbert

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


#  Frequency Domain
fft_values_m = np.fft.fft(msg_signal) / len(msg_signal)
freq_m = np.fft.fftfreq(len(msg_signal), 1 / sample_rate)
magnitude_m = np.abs(fft_values_m) * 2

fft_values_r = np.fft.fft(recovered) / len(recovered)
freq_r = np.fft.fftfreq(len(recovered), 1 / sample_rate)
magnitude_r = np.abs(fft_values_r) * 2


error = np.mean(np.abs(msg_signal - recovered))
print("Mean absolute error:", error) # error = 0.02583321185326483


# Plotting
def plotting(x_axis, y_axis, title, x_axis_title, y_axis_title):
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.grid(True)

start_time = 1.01
end_time = 1.02

start = int(start_time * sample_rate)
end = int(end_time * sample_rate)

# Message signal, FM signal, Recovered signal
plt.figure(figsize = (12, 10))
plt.subplot(3, 1, 1)
plotting(time, msg_signal, "Message Signal", "Time", "Amplitude")
plt.subplot(3, 1, 2)
plotting(time, fm_signal, "FM Signal", "Time", "Amplitude")
plt.subplot(3, 1, 3)
plotting(time, recovered, "Recovered Signal", "Time", "Amplitude")

# Message signal and Recovered signal (zoomed)
plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(time[start:end], msg_signal[start:end], "Message Signal (Zoomed)", "Time", "Amplitude")
plt.subplot(2, 1, 2)
plotting(time[start:end], recovered[start:end], "Recovered Signal (Zoomed)", "Time", "Amplitude")

# Plotting frequency domain - Message signal and Recovered signal
plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(freq_m, magnitude_m, "Message Signal (freq domain)", "Frequency", "Amplitude")
plt.subplot(2, 1, 2)
plotting(freq_r, magnitude_r, "Recovered Signal (freq domain)", "Frequency", "Amplitude")

# Noisy FM signal org, zoomed
plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(time, noisy_fm_signal, "Noisy Signal", "Time", "Amplitude")
plt.subplot(2, 1, 2)
plotting(time[start:end], noisy_fm_signal[start:end], "Noisy Signal (Zoomed)", "Time", "Amplitude")

# Recovered Signal with noise and without
plt.figure(figsize = (18, 8))
plt.subplot(2, 2, 1)
plotting(time, recovered, "Recovered Message Signal", "Time", "Amplitude")
plt.subplot(2, 2, 2)
plotting(time[start:end], recovered[start:end], "Recovered Message Signal (Zoomed)", "Time", "Amplitude")
plt.subplot(2, 2, 3)
plotting(time, recovered_noisy, "Recovered Noisy Signal", "Time", "Amplitude")
plt.subplot(2, 2, 4)
plotting(time[start:end], recovered_noisy[start:end], "Recovered Noisy Signal (Zoomed)", "Time", "Amplitude")

plt.tight_layout()
plt.show()