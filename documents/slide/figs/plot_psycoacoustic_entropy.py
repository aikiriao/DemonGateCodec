'''
知覚エントロピーのプロット
'''
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('psyco_model')
from check_psyco_model import get_frame, PsycoAcousticsModelII

if __name__ == '__main__':
    # 波形データ読み込み
    SAMPLING_FREQUENCY, indata = wavfile.read(sys.argv[1])
    NUM_SAMPLES = indata.shape[0]
    indata = indata.reshape((NUM_SAMPLES, -1))
    NUM_CHANNELS = indata.shape[1]

    model = [PsycoAcousticsModelII(SAMPLING_FREQUENCY) for _ in range(NUM_CHANNELS)]

    pe_list = []
    ch = 0
    for gr in range(NUM_SAMPLES // model[0].NUM_GRANULE_SAMPLES):
        frame = get_frame(model[ch], indata.T[ch], gr)

        # 聴覚心理モデルII計算
        pe, block_type, ratio = model[ch].compute_psyco_model_II(frame)
        pe_list.append(pe)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)

    t = np.arange(len(indata.T[ch])) / SAMPLING_FREQUENCY
    ax1.plot(t, indata.T[ch] / np.iinfo(indata.T[ch][0]).max,
            label='Wave form', linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim((0, max(t)))
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(2, 1, 2)

    ax2.axhline(model[ch].PERCEPTUAL_ENTROPY_THRESHOLD,\
        color='orange', linestyle='--', label='PE threshold')
    ax2.plot(np.arange(len(pe_list)) * model[ch].NUM_GRANULE_SAMPLES / SAMPLING_FREQUENCY,\
        pe_list, color='red', label='Perceptual Entropy (PE)')
    ax2.set_xlabel('time (sec)')
    ax2.set_ylabel('Perceptual entropy')
    ax2.set_xlim((0, max(t)))
    ax2.legend()
    ax2.grid()

    fig.tight_layout()
    plt.savefig('perceptual_entropy_computation_example.pdf')
