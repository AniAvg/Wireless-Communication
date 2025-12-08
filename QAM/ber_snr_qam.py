import numpy as np
import matplotlib.pyplot as plt
import string
import random

def generate_bits(min_len=32, max_len=1024):
    L = np.random.randint(min_len, max_len + 1)
    L += (-L) % 4

    random_bits = np.random.randint(0, 2, size=L, dtype=np.uint8)
    return random_bits


def qam16_mod(bits):
    bits = bits.reshape(-1, 4)
    mapping = {
        (0,0,0,0): -3-3j, (0,0,0,1): -3-1j,
        (0,0,1,0): -3+3j, (0,0,1,1): -3+1j,
        (0,1,0,0): -1-3j, (0,1,0,1): -1-1j,
        (0,1,1,0): -1+3j, (0,1,1,1): -1+1j,
        (1,0,0,0): 3-3j,  (1,0,0,1): 3-1j,
        (1,0,1,0): 3+3j,  (1,0,1,1): 3+1j,
        (1,1,0,0): 1-3j,  (1,1,0,1): 1-1j,
        (1,1,1,0): 1+3j,  (1,1,1,1): 1+1j
    }
    return np.array([mapping[tuple(b)] for b in bits])

def qam16_demod(symbols):
    all_bits = np.array([
        [0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
        [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
        [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1],
        [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]
    ])

    const_points = np.array([
        -3-3j, -3-1j, -3+3j, -3+1j,
        -1-3j, -1-1j, -1+3j, -1+1j,
         3-3j,  3-1j,  3+3j,  3+1j,
         1-3j,  1-1j,  1+3j,  1+1j,
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
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)
    )
    return signal + noise

def simulate():

    snr_list = [0, 5, 10, 15, 20, 25]
    ber_list = []

    for snr in snr_list:
        errors = 0
        total = 0
        for _ in range(1000):
            bits = generate_bits()
            tx = qam16_mod(bits)
            rx = add_awgn(tx, snr)
            decoded_bits = qam16_demod(rx)

            errors += np.sum(bits != decoded_bits)
            total += len(bits)

        ber = errors / total
        ber_list.append(ber)
        print(f"SNR={snr} dB → BER={ber:.6f}")

    return snr_list, ber_list

snrs, bers = simulate()

plt.figure()
plt.semilogy(snrs, bers, marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR (16-QAM)")
plt.grid(True)
plt.show()
