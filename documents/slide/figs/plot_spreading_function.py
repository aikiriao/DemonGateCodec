'''
MP3の広がり関数とSchroederの広がり関数を比較してプロット
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('psyco_model')
from check_psyco_model import  _compute_spreading_function, PARTITION_DATA

def _schroeder_spreading_function_core(delta_bark):
    '''
    Schroederの広がり関数コア
    '''
    offset_bark = delta_bark + 0.474
    return 15.81 + 7.5 * offset_bark - 17.5 * (1.0 + offset_bark ** 2) ** 0.5

def _schroeder_spreading_function(partition):
    '''
    Schroederの広がり関数計算
    '''
    part_max = len(partition)
    spread_func = np.zeros((part_max, part_max))
    for i in range(part_max):
        for j in range(part_max):
            delta_bark = partition[i]['bval'] - partition[j]['bval']
            spread_func[i][j] = _schroeder_spreading_function_core(delta_bark)
    return spread_func

if __name__ == '__main__':
    parts = PARTITION_DATA['long44100']
    barks = [part['bval'] for part in parts]

    sfunc =  _compute_spreading_function(parts)
    schroeder_sfunc = _schroeder_spreading_function(parts)

    # マスカーとマスキーの関係が反転した状態にあるため転置
    # delta = マスキー - マスカーなので、delta > 0、つまりマスキングカーブのピークより後ろでマスキー > マスカーとなり、マスカーの周波数がマスキーより低くなる
    sfunc = sfunc.T
    schroeder_sfunc = schroeder_sfunc.T

    ax = plt.gca()
    for plot_i in (10, 20, 30):
        plot_bark = parts[plot_i]['bval']

        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(barks, 10 * np.log10(sfunc[plot_i]),\
                label=f'MP3 bark:{plot_bark:.2f}', color=color)
        plt.plot(barks, schroeder_sfunc[plot_i],\
                label=f'Schroeder bark:{plot_bark:.2f}', color=color, linestyle='--')

    plt.grid()
    plt.ylim((-70, 5))
    plt.xlabel('Bark scale')
    plt.ylabel('Gain (dB)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('spreading_functons.pdf')
