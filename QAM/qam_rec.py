import numpy as np
import matplotlib.pyplot as plt

msg = input("Enter a message: ")
msg_binary = ""

for c in msg:
    binary = format(ord(c), '08b')
    msg_binary += binary

print("Binary message: ", msg_binary)

msg_symbols = np.array(list(msg_binary), dtype = int)

bits_per_symbol = 4
msg_matrix = np.reshape(msg_symbols, (-1, bits_per_symbol))


mapping = {
        '0000': -3 - 3j, '0001': -3 - 1j, '0011': -3 + 1j, '0010': -3 + 3j,
        '0100': -1 - 3j, '0101': -1 - 1j, '0111': -1 + 1j, '0110': -1 + 3j,
        '1100':  1 - 3j, '1101':  1 - 1j, '1111':  1 + 1j, '1110':  1 + 3j,
        '1000':  3 - 3j, '1001':  3 - 1j, '1011':  3 + 1j, '1010':  3 + 3j,
    }

symbols_string = ["".join(row.astype(str)) for row in msg_matrix]

qam_symbols = np.array([mapping[s] for s in symbols_string])
print("QAM symbols: ", qam_symbols)


def adding_awgn(symbols, snr_db):
    signal_power = np.mean(np.abs(symbols ** 2))
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    std_dev = np.sqrt(noise_power / 2)
    noise = std_dev * (np.random.randn(len(symbols)) +
                   1j * np.random.randn(len(symbols)))
    noisy_symbols = symbols + noise
    return noisy_symbols

qam_noisy_symbols = adding_awgn(qam_symbols, 15)

plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), s = 100)
plt.axhline(0, color='gray', linewidth=1.2)
plt.axvline(0, color='gray', linewidth=1.2)
plt.xlim([-3.5, 3.5])
plt.ylim([-3.5, 3.5])
plt.title("16 QAM Constellation Diagram")
plt.xlabel("In phase")
plt.ylabel("Quadrature")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(np.real(qam_noisy_symbols), np.imag(qam_noisy_symbols), s = 100)
plt.axhline(0, color='gray', linewidth=1.2)
plt.axvline(0, color='gray', linewidth=1.2)
plt.xlim([-3.5, 3.5])
plt.ylim([-3.5, 3.5])
plt.title("16 QAM Constellation Diagram with Noise")
plt.xlabel("In phase")
plt.ylabel("Quadrature")
plt.grid(True)

plt.tight_layout()
plt.show()
