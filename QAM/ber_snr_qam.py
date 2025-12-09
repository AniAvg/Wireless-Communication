import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def generate_bits(n, min_len=100, max_len=2000):
    L = np.random.randint(min_len, max_len + 1)
    L += (-L) % n

    random_bits = np.random.randint(0, 2, size=L, dtype=np.uint8)
    return random_bits

# Theoretical BER
def ber_qam_theoretical(snr_db, M):
    k = np.log2(M)
    snr_lin = 10**(snr_db / 10)
    return (4/k) * (1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt((3 * k / (2*(M-1))) * snr_lin))

# 4 QAM
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

# 16 QAM
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

def simulate_4qam():
    snr_list = [0, 5, 10, 15, 20, 25, 30, 35]
    ber_list = []
    for snr in snr_list:
        errors = 0
        total = 0
        for _ in range(10000):
            bits = generate_bits(2)
            tx = qam4_mod(bits)
            rx = add_awgn(tx, snr)
            decoded_bits = qam4_demod(rx)

            errors += np.sum(bits != decoded_bits)
            total += len(bits)

        ber = errors / total
        ber_list.append(ber)
        print(f"SNR_4={snr} dB → BER_4={ber:.6f}")
    return snr_list, ber_list


def simulate_16qam():

    snr_list = [0, 5, 10, 15, 20, 25, 30, 35]
    ber_list = []
    for snr in snr_list:
        errors = 0
        total = 0
        for _ in range(10000):
            bits = generate_bits(4)
            tx = qam16_mod(bits)
            rx = add_awgn(tx, snr)
            decoded_bits = qam16_demod(rx)

            errors += np.sum(bits != decoded_bits)
            total += len(bits)

        ber = errors / total
        ber_list.append(ber)
        print(f"SNR_16={snr} dB → BER_16={ber:.6f}")
    return snr_list, ber_list

snrs_4qam, bers_4qam = simulate_4qam()
snrs_16qam, bers_16qam = simulate_16qam()

snr_range = np.arange(0, 36, 1)
ber_4qam_th = ber_qam_theoretical(snr_range, 4)
ber_16qam_th = ber_qam_theoretical(snr_range, 16)

plt.figure(figsize = (12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(snrs_4qam, bers_4qam, marker='o', label="4-QAM")
plt.semilogy(snrs_16qam, bers_16qam, marker='o', label="16-QAM")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR (4-QAM and 16-QAM)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(snr_range, ber_4qam_th, 'b-', label="4-QAM Theo")
plt.semilogy(snr_range, ber_16qam_th, 'r-', label="16-QAM Theo")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR (4-QAM and 16-QAM)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
