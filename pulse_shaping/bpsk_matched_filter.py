import numpy as np
import matplotlib.pyplot as plt

num_symbols = 10
sps = 8 #sps -> samples per symbol
beta = 0.35
span = 6

bits = np.random.randint(0, 2, num_symbols)
symbols = 2 * bits - 1

time = np.arange(-span/2, span/2 + 1/sps, 1/sps)
Ts = 1

unsampled = np.zeros(len(symbols) * sps)
unsampled[::sps] = symbols

def raised_cosine(t, Ts, alpha):
    rc = np.zeros_like(t)
    for i in range(len(t)):
        if abs(1 - (2*alpha*t[i]/Ts)**2) < 1e-6:
            rc[i] = np.pi/4 * np.sinc(1/(2*alpha))
        else:
            rc[i] = np.sinc(t[i]/Ts) * \
                    np.cos(np.pi*alpha*t[i]/Ts) / \
                    (1 - (2*alpha*t[i]/Ts)**2)
    return rc


rc_filter = raised_cosine(time, Ts, beta)

tx = np.convolve(unsampled, rc_filter)

matched_filter = rc_filter[::-1]
y = np.convolve(tx, matched_filter)

delay = (len(rc_filter) - 1) // 2
total_delay = 2 * delay

sample_indices = total_delay + np.arange(len(symbols)) * sps
samples = y[sample_indices]
detected = np.sign(samples)


print("Symbols:  ", symbols)
print("Detected: ", detected)


