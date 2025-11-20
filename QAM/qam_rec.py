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

plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), s = 100)
plt.axhline(0, color='gray', linewidth=0.7)
plt.axvline(0, color='gray', linewidth=0.7)
plt.title("16 QAM Constellation Diagram")
plt.xlabel("In phase")
plt.ylabel("Quadrature")
plt.grid(True)
plt.show()

