import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import hilbert

msg_signal, sample_rate = sf.read("original_audio.mp3")

# checking if the audio is mono
if msg_signal.ndim > 1:
    msg_signal = msg_signal[:, 0]

msg_signal = msg_signal / np.max(np.abs(msg_signal))
# T = time step
T = 1 / sample_rate
freq = np.fft.fftfreq(len(msg_signal), T)
Am = np.max(np.abs(msg_signal))

carrier_freq = 5 * np.max(np.abs(freq))
# carrier_freq = 10000
Ac = 1
time = np.arange(len(msg_signal)) / sample_rate

carrier_signal = Ac * np.cos(2 * np.pi * carrier_freq * time)
# s(t) = (Ac + k * m(t)) * cos(2 * pi * fc * t)
mod_index = Am / Ac
am_signal = (Ac + mod_index * msg_signal) * carrier_signal


analytic_signal = hilbert(am_signal)
envelope = np.abs(analytic_signal)
recovered = (envelope / Ac - 1) / mod_index
recovered = np.clip(recovered, -1, 1)

sf.write("recovered_am_audio.mp3", recovered, sample_rate)


#  SNR = P_signal / P_noise

def adding_awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power =signal_power / snr_linear
    std_dev = np.sqrt(noise_power)
    noise = std_dev * np.random.randn(len(signal))
    noisy_signal = signal + noise
    return noisy_signal

noisy_am_signal = adding_awgn(am_signal, 30)

analytic_signal = hilbert(noisy_am_signal)
envelope = np.abs(analytic_signal)
recovered_noisy = (envelope / Ac - 1) / mod_index
recovered_noisy = np.clip(recovered_noisy, -1, 1)

sf.write("recovered_noisy_am_audio.mp3", recovered_noisy, sample_rate)

#  Frequency Domain
fft_values_m = np.fft.fft(msg_signal) / len(msg_signal)
freq_m = np.fft.fftfreq(len(msg_signal), 1 / sample_rate)
magnitude_m = np.abs(fft_values_m) * 2

fft_values_r = np.fft.fft(recovered) / len(recovered)
freq_r = np.fft.fftfreq(len(recovered), 1 / sample_rate)
magnitude_r = np.abs(fft_values_r) * 2


error = np.mean(np.abs(msg_signal - recovered))
print("Mean absolute error:", error) # error = 0.006915288466805749

# Plotting
def plotting(x_axis, y_axis, title, x_axis_title, y_axis_title):
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.grid(True)

start_time = 0
end_time = 0.01

start = int(start_time * sample_rate)
end = int(end_time * sample_rate)

# Msg signal, AM signal, Recovered signal
plt.figure(figsize = (12, 8))
plt.subplot(3, 1, 1)
plotting(time, msg_signal, "Message Signal", "Time", "Amplitude")
plt.subplot(3, 1, 2)
plotting(time, am_signal, "AM Signal", "Time", "Amplitude")
plt.subplot(3, 1, 3)
plotting(time, recovered, "Recovered Signal", "Time", "Amplitude")

# Msg signal, Recovered signal (zoomed)
plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(time[start:end], msg_signal[start:end], "Message Signal (Zoomed)", "Time", "Amplitude")
plt.subplot(2, 1, 2)
plotting(time[start:end], recovered[start:end], "Recovered Signal (Zoomed)", "Time", "Amplitude")

# Plotting frequency domain - Msg signal and Recovered signal
plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(freq_m, magnitude_m, "Message Signal (freq domain)", "Frequency", "Amplitude")
plt.subplot(2, 1, 2)
plotting(freq_r, magnitude_r, "Recovered Signal (freq domain)", "Frequency", "Amplitude")

# Noisy signal
plt.figure(figsize = (10, 8))
plt.subplot(2, 1, 1)
plotting(time, noisy_am_signal, "Noisy Signal", "Time", "Amplitude")
plt.subplot(2, 1, 2)
plotting(time[start:end], noisy_am_signal[start:end], "Noisy Signal (Zoomed)", "Time", "Amplitude")

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