import numpy as np
import matplotlib.pyplot as plt

def qam4_mod(bits):
    bits = bits.reshape(-1, 2)
    mapping = {
        (0, 0): -1-1j,
        (0, 1): -1+1j,
        (1, 0): 1-1j,
        (1, 1): 1+1j
    }
    return np.array([mapping[tuple(b)] for b in bits])

def qam4_demod(symbols):
    all_bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    const_points = np.array([
        -1-1j, -1+1j, 1-1j, 1+1j
    ])

    decoded_bits = []
    for s in symbols:
        idx = np.argmin(np.abs(const_points - s))
        decoded_bits.append(all_bits[idx])
    return np.vstack(decoded_bits).flatten()

def add_awgn(signal, snr_db):
    snr_lin = 10**(snr_db / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_lin
    noise = (np.sqrt(noise_power/2) *
            (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)))
    return signal + noise

N = 64
cp_len = N // 8
snr_db = 20

bits_number = 10000
bits = np.random.randint(0, 2, size=bits_number)
if len(bits) % 2 != 0:
    bits = np.append(bits, 0)

qam_symbols = qam4_mod(bits)

# Serial to parallel
num_symbols = int(np.ceil(len(qam_symbols) / N))
pad_len = num_symbols * N - len(qam_symbols)
if pad_len > 0:
    symbols = np.concatenate([qam_symbols, np.zeros(pad_len, dtype=np.complex128)])

ofdm_blocks = symbols.reshape(num_symbols, N)

time_domain = np.fft.ifft(ofdm_blocks, n=N, axis=1)


