'''
パーティションの周波数分割をプロット
'''
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('psyco_model')
from check_psyco_model import PsycoAcousticsModelII

def _compute_partition_frequencies(partition, sampling_frequency, window_size):
    '''
    パーティションの最小・最大周波数を計算

    Parameters
    ----------
    partition : list of dict
        パーティション
    sampling_frequency : int
        サンプリングレート
    window_size : int
        フレーム窓サイズ

    Returns
    -------
    min_freqs : ndarray
        パーティションの最小周波数
    max_freqs : ndarray
        パーティションの最大周波数
    '''
    min_freqs = np.zeros(len(partition))
    max_freqs = np.zeros(len(partition))
    index = 0
    for part_index, part in enumerate(partition):
        min_freqs[part_index] = index / window_size * sampling_frequency
        max_freqs[part_index] = (index + part['#lines']) / window_size * sampling_frequency
        index += part['#lines']
    return min_freqs, max_freqs

if __name__ == "__main__":
    model = PsycoAcousticsModelII(44100)
    xlist = np.arange(len(model.PARTITION_LONG))
    min_freqs, max_freqs = _compute_partition_frequencies(
            model.PARTITION_LONG, model.SAMPLING_FREQUENCY, model.LONG_WINDOW_SIZE)
    for inx, psy in enumerate(model.PSYCO_LONG):
        bu = psy['bu']
        bo = psy['bo']
        plt.axvline(x = bu, linestyle = '--', linewidth = 0.5, alpha = 0.8)
        plt.axvline(x = bo, linestyle = '--', linewidth = 0.5, alpha = 0.8)
        plt.text((bu + bo) / 2.0, max_freqs[bo] + 500, f'{inx + 1}', fontsize=10, ha='center')
    plt.plot(min_freqs, label='min partition frequency', color='blue')
    plt.plot(max_freqs, label='max partition frequency', color='red')
    # for i, x in enumerate(xlist):
        # plt.vlines(x=x, ymin=min_freqs[i], ymax=max_freqs[i], linewidth=2.0)
    plt.xlabel('Partition number')
    plt.ylabel('Frequency (Hz)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('partion_frequency_long44100Hz.pdf')
