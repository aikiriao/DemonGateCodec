import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from itertools import cycle

def mdct(indata):
    N = len(indata) // 2
    out = np.zeros(N)
    for n in range(2 * N):
        for k in range(N):
            out[k] += indata[n] * np.cos(np.pi / N * ( k + 1/2 ) * ( n + 1/2 + N/2 ))
    return out

def imdct(inspec):
    N = len(inspec)
    out = np.zeros(2 * N)
    for k in range(N):
        for n in range(2 * N):
            out[n] += inspec[k] * np.cos(np.pi / N * ( k + 1/2 ) * ( n + 1/2 + N/2 ))
    out *= 2/N
    return out

if __name__ == "__main__":
    LINES = ['-', '--', '-.', ':']
    LINECYCLER = cycle(LINES)

    FRAME_SIZE = 36
    SLIDE_SIZE = 18
    NUM_SAMPLES = FRAME_SIZE * 5

    samples = np.arange(0, NUM_SAMPLES)
    data = np.sin(2.0 * np.pi * samples / 30.0)

    WINDOW = np.sin((np.arange(0, FRAME_SIZE) + 1/2) * np.pi / FRAME_SIZE)

    smpl = 0
    prev_frame = np.zeros(FRAME_SIZE // 2)
    decoded_data = np.zeros_like(data)
    while smpl + FRAME_SIZE <= NUM_SAMPLES:
        inframe = data[smpl:smpl + FRAME_SIZE].copy()
        inframe *= WINDOW
        inspec = mdct(inframe)
        decframe = imdct(inspec)
        decframe *= WINDOW
        plt.subplots(figsize=(6, 3))
        plt.axhline(0.0, color='k')
        plt.plot(inframe, label='input', linewidth=3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'mdct_frame_waveform_{smpl}.png', transparent=True)
        plt.close()
        plt.subplots(figsize=(6, 3))
        plt.axhline(0.0, color='k')
        plt.plot(inspec, label='MDCT(input)', linewidth=3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'mdct_frame_spectrum_{smpl}.png', transparent=True)
        plt.close()
        plt.subplots(figsize=(6, 3))
        plt.axhline(0.0, color='k')
        plt.axvline(SLIDE_SIZE, color='k', linestyle='--')
        plt.plot(decframe, label='IMDCT(MDCT(input))', linewidth=3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'mdct_decoded_waveform_{smpl}.png', transparent=True)
        plt.close()
        decoded_data[smpl:smpl + SLIDE_SIZE] = prev_frame + decframe[:FRAME_SIZE // 2]
        prev_frame = decframe[FRAME_SIZE // 2:]
        smpl += SLIDE_SIZE

    plt.subplots(figsize=(27, 3))
    plt.axhline(0.0, color='k')
    for smpl in np.arange(0, NUM_SAMPLES - SLIDE_SIZE, SLIDE_SIZE):
        plt.axvline(smpl, color='k', linestyle='--')
    plt.plot(data[SLIDE_SIZE:-SLIDE_SIZE], linewidth=3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'mdct_input_waveform.png')
    plt.close()

    plt.subplots(figsize=(27, 3))
    plt.axhline(0.0, color='k')
    for smpl in np.arange(0, NUM_SAMPLES - SLIDE_SIZE, SLIDE_SIZE):
        plt.axvline(smpl, color='k', linestyle='--')
    plt.plot(decoded_data[SLIDE_SIZE:-SLIDE_SIZE], linewidth=3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'mdct_decoded_waveform.png')
