import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from itertools import cycle
from mp3_routines import ENCODER_FILTER_COEF, DECODER_FILTER_COEF

if __name__ == "__main__":
    LINES = ['-', '--', '-.', ':']
    LINECYCLER = cycle(LINES)

    # 係数プロット
    plt.cla()
    plt.plot(ENCODER_FILTER_COEF)
    plt.title('MP3 Encoder prototype filter coefficients')
    plt.grid()
    plt.xlabel('index')
    plt.ylabel('amplitude')
    plt.tight_layout()
    plt.savefig('mp3_encoder_prototype_filter_coef.pdf')
    plt.cla()
    plt.plot(DECODER_FILTER_COEF)
    plt.title('MP3 Decoder prototype filter coefficients')
    plt.grid()
    plt.xlabel('index')
    plt.ylabel('amplitude')
    plt.tight_layout()
    plt.savefig('mp3_decoder_prototype_filter_coef.pdf')

    # 周波数特性
    bank_specs = []
    for k in range(32):
        coef = ENCODER_FILTER_COEF * np.cos(np.pi / 32 * (k + 1/2) * (np.arange(0, 512) - 16))
        spec = np.fft.fft(coef, norm='forward')[:len(coef)//2]
        bank_specs.append(spec)

    plt.cla()
    BANK_FREQ = np.fft.fftfreq(len(ENCODER_FILTER_COEF), 1.0 / (2.0 * np.pi))[:len(ENCODER_FILTER_COEF)//2]
    for k in np.arange(0, 16):
        bank_spec = bank_specs[k]
        linetype = next(LINECYCLER)
        plt.plot(BANK_FREQ, 20.0 * np.log10(np.abs(bank_spec)), linestyle=linetype, label=f'Bank {k}')
    plt.xlabel('normalized frequency')
    plt.ylabel('amplitude')
    plt.xlim((0, np.pi))
    plt.legend(loc='upper right', ncols=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig('mp3_encoder_filter_bank_frequency_spec_0_15.pdf')

    plt.cla()
    BANK_FREQ = np.fft.fftfreq(len(ENCODER_FILTER_COEF), 1.0 / (2.0 * np.pi))[:len(ENCODER_FILTER_COEF)//2]
    for k in np.arange(16, 32):
        bank_spec = bank_specs[k]
        linetype = next(LINECYCLER)
        plt.plot(BANK_FREQ, 20.0 * np.log10(np.abs(bank_spec)), linestyle=linetype, label=f'Bank {k}')
    plt.xlabel('normalized frequency')
    plt.ylabel('amplitude')
    plt.xlim((0, np.pi))
    plt.legend(loc='upper left', ncols=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig('mp3_encoder_filter_bank_frequency_spec_16_31.pdf')

    # 電力相補条件のチェック
    plt.cla()
    for k in range(32):
        gk_coef = ENCODER_FILTER_COEF[k::64]
        denom = [0.0] * 8
        denom[7] = 1.0
        w, gkz = signal.freqz(gk_coef, a = 1.0)
        _, gkinvz = signal.freqz(gk_coef[::-1], a = denom)
        gkM_coef = ENCODER_FILTER_COEF[k + 32::64]
        _, gkMz = signal.freqz(gkM_coef, a = 1.0)
        _, gkMinvz = signal.freqz(gkM_coef[::-1], a = denom)
        Gk = gkz * gkinvz + gkMz * gkMinvz
        linetype = next(LINECYCLER)
        plt.plot(w, np.real(Gk), label=f'real', linestyle=linetype)
        plt.plot(w, np.imag(Gk), label=f'imag', linestyle=linetype)
    plt.annotate('Real part', xy=(np.pi / 2, 1.0 / 512), xytext=(0.5, 1.0 / 1024), arrowprops=dict(arrowstyle='->', facecolor='black'))
    plt.annotate('Imaginary part', xy=(np.pi / 2, 0.0), xytext=(2.0, 1.0 / 1024), arrowprops=dict(arrowstyle='->', facecolor='black'))
    plt.title('MP3 prototype filter power complementary condition check')
    plt.grid()
    plt.xlabel('normalized frequency')
    plt.ylabel('amplitude')
    plt.tight_layout()
    plt.savefig('mp3_encoder_prototype_filter_power_complementary_condition.pdf')
