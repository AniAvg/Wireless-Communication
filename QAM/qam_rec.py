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

i_bits = msg_matrix[: len(msg_matrix)//2]
q_bits = msg_matrix[len(msg_matrix)//2 :]

mapping_table = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): 1,
    (1, 0): 3
}

I = mapping_table[i_bits]
Q = mapping_table[q_bits]

