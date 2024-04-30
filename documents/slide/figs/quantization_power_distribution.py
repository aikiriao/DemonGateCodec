import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    LINES = ['-', '--', '-.', ':']

    data = np.abs(np.random.laplace(0.0, 0.1, 2 ** 15))
    powered = np.sign(data) * (np.abs(data) ** (3.0 / 4.0))

    data_freq, edges = np.histogram(data, bins=400, density=True)
    powered_freq, _ = np.histogram(powered, bins=400, density=True)

    bin_width = edges[1] - edges[0]
    plt.plot(edges[:-1], data_freq * bin_width, label='|x|')
    plt.plot(edges[:-1], powered_freq * bin_width, label='|x|^(3/4)')
    plt.ylabel('Frequency')
    plt.xlabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('powered_distribution.pdf')
