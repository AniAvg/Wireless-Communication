import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def generate_bits(n, min_len=100, max_len=2000):
    L = np.random.randint(min_len, max_len + 1)
    L += (-L) % n
    return np.random.randint(0, 2, size=L, dtype=np.uint8)

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
    symbols = np.array([mapping[tuple(b)] for b in bits])
    return symbols / np.sqrt(2)

def qam4_demod(symbols):
    all_bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    const_points = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2)

    decoded_bits = []
    for s in symbols:
        idx = np.argmin(np.abs(const_points - s))
        decoded_bits.append(all_bits[idx])
    return np.hstack(decoded_bits)

# 16 QAM
def qam16_mod(bits):
    bits = bits.reshape(-1, 4)
    gray_map = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}

    symbols = []
    for b in bits:
        I = gray_map[(b[0], b[1])]
        Q = gray_map[(b[2], b[3])]
        symbols.append(I + 1j * Q)
    symbols = np.array(symbols)
    return symbols / np.sqrt(10)

# def qam16_demod(symbols):
#     levels = np.array([-3, -1, 1, 3])
#     gray_bits = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}
#
#     bits_out = []
#     for s in symbols:
#         I_hat = levels[np.argmin(np.abs(levels - s.real))]
#         Q_hat = levels[np.argmin(np.abs(levels - s.imag))]
#
#         bits_out.extend(gray_bits[I_hat * np.sqrt(10)])
#         bits_out.extend(gray_bits[Q_hat * np.sqrt(10)])
#     return np.array(bits_out, dtype=np.uint8)

def qam16_demod(symbols):
    thresh = np.array([-2, 0, 2]) / np.sqrt(10)

    def decide(x):
        if x < thresh[0]: return -3
        if x < thresh[1]: return -1
        if x < thresh[2]: return 1
        return 3

    gray_bits = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}

    bits_out = []
    for s in symbols:
        I = decide(s.real)
        Q = decide(s.imag)
        bits_out.extend(gray_bits[I])
        bits_out.extend(gray_bits[Q])
    return np.array(bits_out, dtype=np.uint8)


def add_awgn(signal, snr_db):
    snr_lin = 10**(snr_db / 10)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / snr_lin
    noise = (np.sqrt(noise_power / 2) *
            (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)))
    return signal + noise

# def energy_info(symbols, M):
#     k = np.log2(M)
#     Es = np.mean(np.abs(symbols)**2)
#     Eb = Es / k
#     return Es, Eb, k

def simulate(mod, demod, bits_per_symbol, M):
    snr_list = np.arange(0, 36, 5)  # 0 to 35 dB
    ber_list = []

    for snr in snr_list:
        errors = 0
        total_bits = 0
        for _ in range(3000):
            bits = generate_bits(bits_per_symbol)
            tx = mod(bits)
            rx = add_awgn(tx, snr)
            rx_bits = demod(rx)

            errors += np.sum(bits != rx_bits)
            total_bits += len(bits)

        ber = errors / total_bits if total_bits > 0 else 1
        ber_list.append(ber)
        print(f"{M}-QAM  SNR={snr:2d} dB → BER={ber:.2e}")

    return snr_list, ber_list


print("4-QAM Simulation")
snrs_4qam, bers_4qam = simulate(qam4_mod, qam4_demod, 2, 4)
print("\n16-QAM Simulation")
snrs_16qam, bers_16qam = simulate(qam16_mod, qam16_demod, 4, 16)

snr_fine = np.linspace(0, 35, 200)
ber_4qam_th = ber_qam_theoretical(snr_fine, 4)
ber_16qam_th = ber_qam_theoretical(snr_fine, 16)

plt.figure(figsize = (10, 6))
plt.semilogy(snrs_4qam, bers_4qam, marker='o', label="4-QAM")
plt.semilogy(snrs_16qam, bers_16qam, marker='o', label="16-QAM")
plt.semilogy(snr_fine, ber_4qam_th, '-', label='4-QAM Theoretical', linewidth=2)
plt.semilogy(snr_fine, ber_16qam_th, '--', label='16-QAM Theoretical', linewidth=2)
plt.xlabel("Eₛ/N₀ (dB)")
plt.ylabel("BER")
plt.title("BER vs Eₛ/N₀ (4-QAM and 16-QAM)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.ylim(1e-7, 1)

# Trueplt.figure(figsize=(7,4))
# plt.bar(['4-QAM Es','4-QAM Eb','16-QAM Es','16-QAM Eb'],
#         [np.mean(Es4), np.mean(Eb4), np.mean(Es16), np.mean(Eb16)],
#         color=['blue','skyblue','red','salmon'])
# plt.title("Symbol and Bit Energy")
# plt.ylabel("Energy")
# plt.grid(True, axis='y', ls='--')

plt.tight_layout()
plt.show()
