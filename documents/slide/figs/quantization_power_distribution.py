import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    var = 0.1
    scale = (var / 2) ** 0.5

    # data = np.abs(np.random.laplace(0.0, scale, 2 ** 15))
    # powered = np.sign(data) * (np.abs(data) ** (3.0 / 4.0))
    # data_freq, edges = np.histogram(data, bins=400, density=True)
    # powered_freq, _ = np.histogram(powered, bins=400, density=True)
    # bin_width = edges[1] - edges[0]
    # plt.plot(edges[:-1], data_freq * bin_width, label='|x|')
    # plt.plot(edges[:-1], powered_freq * bin_width, label='|x|^(3/4)')
    # plt.ylabel('Frequency')
    # plt.xlabel('Amplitude')
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('powered_distribution.pdf')

    x = np.linspace(0, 2.0, 4096)
    plt.cla()
    laplace = 1.0 / scale * np.exp(- x / scale)
    powered = 4.0 / (3.0 * scale) * (x ** (1 / 3)) * np.exp(- (x ** (4 / 3) / scale))
    plt.plot(x, laplace, label='|x|')
    plt.plot(x, powered, label='|x|^(3/4)')
    plt.ylabel('Density')
    plt.xlabel('|x|')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('powered_distribution.pdf')
