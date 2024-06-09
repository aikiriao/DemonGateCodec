'''
分析結果のプロット
'''
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('psyco_model')
from check_psyco_model import get_frame, PsycoAcousticsModelII

def _plot_analyze_result(model):
    '''
    分析結果のプロット
    '''
    freqs = np.linspace(0, model.SAMPLING_FREQUENCY / 2.0 + 1, len(model.PARTITION_INDEX_LONG))
    eb_long_spec = [model.eb_long[model.PARTITION_INDEX_LONG[i]]\
        for i in range(len(model.PARTITION_INDEX_LONG))]
    etb_long_spec = [model.ecb_long[model.PARTITION_INDEX_LONG[i]]\
        for i in range(len(model.PARTITION_INDEX_LONG))]
    nb_long_spec = [model.nb_long[model.PARTITION_INDEX_LONG[i]]\
        for i in range(len(model.PARTITION_INDEX_LONG))]
    thr_long_spec = [model.thr_long[model.PARTITION_INDEX_LONG[i]]\
        for i in range(len(model.PARTITION_INDEX_LONG))]
    sf_freqs = []
    for psy in model.PSYCO_LONG:
        min_part = -1
        max_part = -1
        for i in range(len(model.PARTITION_INDEX_LONG)):
            part_index = model.PARTITION_INDEX_LONG[i]
            if part_index == psy['bu']:
                min_part = i
            if part_index  == psy['bo']:
                max_part = i
            if min_part >= 0 and max_part >= 0:
                break
        center_freq = 0.5 * (min_part + max_part)\
                * model.SAMPLING_FREQUENCY / (2.0 * len(model.PARTITION_INDEX_LONG))
        sf_freqs.append(center_freq)

    plt.cla()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.xscale('log')
    plt.xlim((50.0, max(freqs)))
    plt.grid()
    plt.plot(freqs, 10 * np.log10(model.energy_long[:len(model.PARTITION_INDEX_LONG)]), label='energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('psyco_analyze_energy.pdf')
    plt.plot(freqs, 10 * np.log10(eb_long_spec), label='partitoned energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('psyco_analyze_partitioned_energy.pdf')
    plt.plot(freqs, 10 * np.log10(etb_long_spec), label='convolved partitoned energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('psyco_analyze_convolved_partitioned_energy.pdf')
    plt.plot(freqs, 10 * np.log10(nb_long_spec), label='noise permissive level')
    plt.legend()
    plt.tight_layout()
    plt.savefig('psyco_analyze_noise_permissive_level.pdf')
    plt.plot(freqs, 10 * np.log10(thr_long_spec), label='threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('psyco_analyze_threshold.pdf')
    plt.cla()
    fig, ax1 = plt.subplots()
    ax1.plot(sf_freqs, 20.0 * np.log10(model.ratio_long), label='SMR', color='red')
    ax1.legend(loc='upper left')
    ax1.set_xscale('log')
    ax1.set_xlim((50.0, max(sf_freqs)))
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Signal-to-Mask Ratio (SMR) (dB)')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (dB)')
    ax2.plot(freqs, 10 * np.log10(eb_long_spec),
        label='partitoned energy', alpha=0.8, linestyle='--')
    ax2.plot(freqs, 10 * np.log10(thr_long_spec),
        label='threshold', alpha=0.8, linestyle='--')
    ax2.legend(loc='upper right')
    ax1.grid()
    fig.tight_layout()
    plt.savefig('psyco_analyze_ratio.pdf')

if __name__ == '__main__':
    # 波形データ読み込み
    SAMPLING_FREQUENCY, indata = wavfile.read(sys.argv[1])
    NUM_SAMPLES = indata.shape[0]
    indata = indata.reshape((NUM_SAMPLES, -1))
    NUM_CHANNELS = indata.shape[1]

    model = [PsycoAcousticsModelII(SAMPLING_FREQUENCY) for _ in range(NUM_CHANNELS)]

    count = 0
    
    for gr in range(NUM_SAMPLES // model[0].NUM_GRANULE_SAMPLES):
        for ch in range(NUM_CHANNELS):
            frame = get_frame(model[ch], indata.T[ch], gr)

            # 聴覚心理モデルII計算
            model[ch].compute_psyco_model_II(frame)

            # 特定カウントでグラフ出力
            if count == 410:
                _plot_analyze_result(model[ch])
                sys.exit(0)

            count += 1
