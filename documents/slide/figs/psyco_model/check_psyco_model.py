'''
聴覚心理モデルIIのチェック
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

PARTITION_DATA = {
    'long48000': [
        { '#lines':  1, 'minval': 24.5, 'qthr':  4.532, 'norm': 0.970, 'bval':  0.000 },
        { '#lines':  1, 'minval': 24.5, 'qthr':  4.532, 'norm': 0.755, 'bval':  0.469 },
        { '#lines':  1, 'minval': 24.5, 'qthr':  4.532, 'norm': 0.738, 'bval':  0.938 },
        { '#lines':  1, 'minval': 24.5, 'qthr':  0.904, 'norm': 0.730, 'bval':  1.406 },
        { '#lines':  1, 'minval': 24.5, 'qthr':  0.904, 'norm': 0.724, 'bval':  1.875 },
        { '#lines':  1, 'minval': 20.0, 'qthr':  0.090, 'norm': 0.723, 'bval':  2.344 },
        { '#lines':  1, 'minval': 20.0, 'qthr':  0.090, 'norm': 0.723, 'bval':  2.813 },
        { '#lines':  1, 'minval': 20.0, 'qthr':  0.029, 'norm': 0.723, 'bval':  3.281 },
        { '#lines':  1, 'minval': 20.0, 'qthr':  0.029, 'norm': 0.718, 'bval':  3.750 },
        { '#lines':  1, 'minval': 20.0, 'qthr':  0.009, 'norm': 0.690, 'bval':  4.199 },
        { '#lines':  1, 'minval': 20.0, 'qthr':  0.009, 'norm': 0.660, 'bval':  4.625 },
        { '#lines':  1, 'minval': 18.0, 'qthr':  0.009, 'norm': 0.641, 'bval':  5.047 },
        { '#lines':  1, 'minval': 18.0, 'qthr':  0.009, 'norm': 0.600, 'bval':  5.438 },
        { '#lines':  1, 'minval': 18.0, 'qthr':  0.009, 'norm': 0.584, 'bval':  5.828 },
        { '#lines':  1, 'minval': 12.0, 'qthr':  0.009, 'norm': 0.532, 'bval':  6.188 },
        { '#lines':  1, 'minval': 12.0, 'qthr':  0.009, 'norm': 0.537, 'bval':  6.522 },
        { '#lines':  2, 'minval':  6.0, 'qthr':  0.018, 'norm': 0.857, 'bval':  7.174 },
        { '#lines':  2, 'minval':  6.0, 'qthr':  0.018, 'norm': 0.858, 'bval':  7.801 },
        { '#lines':  2, 'minval':  3.0, 'qthr':  0.018, 'norm': 0.853, 'bval':  8.402 },
        { '#lines':  2, 'minval':  3.0, 'qthr':  0.018, 'norm': 0.824, 'bval':  8.966 },
        { '#lines':  2, 'minval':  3.0, 'qthr':  0.018, 'norm': 0.778, 'bval':  9.484 },
        { '#lines':  2, 'minval':  3.0, 'qthr':  0.018, 'norm': 0.740, 'bval':  9.966 },
        { '#lines':  2, 'minval':  0.0, 'qthr':  0.018, 'norm': 0.709, 'bval': 10.426 },
        { '#lines':  2, 'minval':  0.0, 'qthr':  0.018, 'norm': 0.676, 'bval': 10.866 },
        { '#lines':  2, 'minval':  0.0, 'qthr':  0.018, 'norm': 0.632, 'bval': 11.279 },
        { '#lines':  2, 'minval':  0.0, 'qthr':  0.018, 'norm': 0.592, 'bval': 11.669 },
        { '#lines':  2, 'minval':  0.0, 'qthr':  0.018, 'norm': 0.553, 'bval': 12.042 },
        { '#lines':  2, 'minval':  0.0, 'qthr':  0.018, 'norm': 0.510, 'bval': 12.386 },
        { '#lines':  2, 'minval':  0.0, 'qthr':  0.018, 'norm': 0.513, 'bval': 12.721 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.608, 'bval': 13.115 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.673, 'bval': 13.562 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.637, 'bval': 13.984 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.586, 'bval': 14.371 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.571, 'bval': 14.741 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.616, 'bval': 15.140 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.640, 'bval': 15.563 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.598, 'bval': 15.962 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.538, 'bval': 16.324 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.512, 'bval': 16.665 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.528, 'bval': 17.020 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.517, 'bval': 17.373 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.493, 'bval': 17.708 },
        { '#lines':  6, 'minval':  0.0, 'qthr':  0.054, 'norm': 0.499, 'bval': 18.045 },
        { '#lines':  7, 'minval':  0.0, 'qthr':  0.063, 'norm': 0.525, 'bval': 18.398 },
        { '#lines':  7, 'minval':  0.0, 'qthr':  0.063, 'norm': 0.541, 'bval': 18.762 },
        { '#lines':  8, 'minval':  0.0, 'qthr':  0.072, 'norm': 0.528, 'bval': 19.120 },
        { '#lines':  8, 'minval':  0.0, 'qthr':  0.072, 'norm': 0.510, 'bval': 19.466 },
        { '#lines':  8, 'minval':  0.0, 'qthr':  0.072, 'norm': 0.506, 'bval': 19.807 },
        { '#lines': 10, 'minval':  0.0, 'qthr':  0.180, 'norm': 0.525, 'bval': 20.159 },
        { '#lines': 10, 'minval':  0.0, 'qthr':  0.180, 'norm': 0.536, 'bval': 20.522 },
        { '#lines': 10, 'minval':  0.0, 'qthr':  0.180, 'norm': 0.518, 'bval': 20.874 },
        { '#lines': 13, 'minval':  0.0, 'qthr':  0.372, 'norm': 0.501, 'bval': 21.214 },
        { '#lines': 13, 'minval':  0.0, 'qthr':  0.372, 'norm': 0.497, 'bval': 21.553 },
        { '#lines': 14, 'minval':  0.0, 'qthr':  0.400, 'norm': 0.497, 'bval': 21.892 },
        { '#lines': 18, 'minval':  0.0, 'qthr':  1.627, 'norm': 0.495, 'bval': 22.231 },
        { '#lines': 18, 'minval':  0.0, 'qthr':  1.627, 'norm': 0.494, 'bval': 22.569 },
        { '#lines': 20, 'minval':  0.0, 'qthr':  1.808, 'norm': 0.497, 'bval': 22.909 },
        { '#lines': 25, 'minval':  0.0, 'qthr': 22.607, 'norm': 0.494, 'bval': 23.248 },
        { '#lines': 25, 'minval':  0.0, 'qthr': 22.607, 'norm': 0.487, 'bval': 23.583 },
        { '#lines': 35, 'minval':  0.0, 'qthr': 31.650, 'norm': 0.483, 'bval': 23.915 },
        { '#lines': 67, 'minval':  0.0, 'qthr': 605.867, 'norm': 0.482, 'bval':24.246 },
        { '#lines': 67, 'minval':  0.0, 'qthr': 605.867, 'norm': 0.524, 'bval':24.576 },
    ],
    'long44100': [
        { '#lines':  1, 'minval': 24.5, 'qthr':   4.532, 'norm': 0.951, 'bval':  0.000 },
        { '#lines':  1, 'minval': 24.5, 'qthr':   4.532, 'norm': 0.700, 'bval':  0.431 },
        { '#lines':  1, 'minval': 24.5, 'qthr':   4.532, 'norm': 0.681, 'bval':  0.861 },
        { '#lines':  1, 'minval': 24.5, 'qthr':   0.904, 'norm': 0.675, 'bval':  1.292 },
        { '#lines':  1, 'minval': 24.5, 'qthr':   0.904, 'norm': 0.667, 'bval':  1.723 },
        { '#lines':  1, 'minval': 20.0, 'qthr':   0.090, 'norm': 0.665, 'bval':  2.153 },
        { '#lines':  1, 'minval': 20.0, 'qthr':   0.090, 'norm': 0.664, 'bval':  2.584 },
        { '#lines':  1, 'minval': 20.0, 'qthr':   0.029, 'norm': 0.664, 'bval':  3.015 },
        { '#lines':  1, 'minval': 20.0, 'qthr':   0.029, 'norm': 0.664, 'bval':  3.445 },
        { '#lines':  1, 'minval': 20.0, 'qthr':   0.029, 'norm': 0.655, 'bval':  3.876 },
        { '#lines':  1, 'minval': 20.0, 'qthr':   0.009, 'norm': 0.616, 'bval':  4.279 },
        { '#lines':  1, 'minval': 20.0, 'qthr':   0.009, 'norm': 0.597, 'bval':  4.670 },
        { '#lines':  1, 'minval': 18.0, 'qthr':   0.009, 'norm': 0.578, 'bval':  5.057 },
        { '#lines':  1, 'minval': 18.0, 'qthr':   0.009, 'norm': 0.541, 'bval':  5.416 },
        { '#lines':  1, 'minval': 18.0, 'qthr':   0.009, 'norm': 0.575, 'bval':  5.774 },
        { '#lines':  2, 'minval': 12.0, 'qthr':   0.018, 'norm': 0.856, 'bval':  6.422 },
        { '#lines':  2, 'minval':  6.0, 'qthr':   0.018, 'norm': 0.846, 'bval':  7.026 },
        { '#lines':  2, 'minval':  6.0, 'qthr':   0.018, 'norm': 0.840, 'bval':  7.609 },
        { '#lines':  2, 'minval':  3.0, 'qthr':   0.018, 'norm': 0.822, 'bval':  8.168 },
        { '#lines':  2, 'minval':  3.0, 'qthr':   0.018, 'norm': 0.800, 'bval':  8.710 },
        { '#lines':  2, 'minval':  3.0, 'qthr':   0.018, 'norm': 0.753, 'bval':  9.207 },
        { '#lines':  2, 'minval':  3.0, 'qthr':   0.018, 'norm': 0.704, 'bval':  9.662 },
        { '#lines':  2, 'minval':  0.0, 'qthr':   0.018, 'norm': 0.674, 'bval': 10.099 },
        { '#lines':  2, 'minval':  0.0, 'qthr':   0.018, 'norm': 0.640, 'bval': 10.515 },
        { '#lines':  2, 'minval':  0.0, 'qthr':   0.018, 'norm': 0.609, 'bval': 10.917 },
        { '#lines':  2, 'minval':  0.0, 'qthr':   0.018, 'norm': 0.566, 'bval': 11.293 },
        { '#lines':  2, 'minval':  0.0, 'qthr':   0.018, 'norm': 0.535, 'bval': 11.652 },
        { '#lines':  2, 'minval':  0.0, 'qthr':   0.018, 'norm': 0.531, 'bval': 11.997 },
        { '#lines':  3, 'minval':  0.0, 'qthr':   0.027, 'norm': 0.615, 'bval': 12.394 },
        { '#lines':  3, 'minval':  0.0, 'qthr':   0.027, 'norm': 0.686, 'bval': 12.850 },
        { '#lines':  3, 'minval':  0.0, 'qthr':   0.027, 'norm': 0.650, 'bval': 13.277 },
        { '#lines':  3, 'minval':  0.0, 'qthr':   0.027, 'norm': 0.612, 'bval': 13.681 },
        { '#lines':  3, 'minval':  0.0, 'qthr':   0.027, 'norm': 0.567, 'bval': 14.062 },
        { '#lines':  3, 'minval':  0.0, 'qthr':   0.027, 'norm': 0.520, 'bval': 14.411 },
        { '#lines':  3, 'minval':  0.0, 'qthr':   0.027, 'norm': 0.513, 'bval': 14.751 },
        { '#lines':  4, 'minval':  0.0, 'qthr':   0.036, 'norm': 0.557, 'bval': 15.119 },
        { '#lines':  4, 'minval':  0.0, 'qthr':   0.036, 'norm': 0.584, 'bval': 15.508 },
        { '#lines':  4, 'minval':  0.0, 'qthr':   0.036, 'norm': 0.570, 'bval': 15.883 },
        { '#lines':  5, 'minval':  0.0, 'qthr':   0.045, 'norm': 0.579, 'bval': 16.263 },
        { '#lines':  5, 'minval':  0.0, 'qthr':   0.045, 'norm': 0.585, 'bval': 16.654 },
        { '#lines':  5, 'minval':  0.0, 'qthr':   0.045, 'norm': 0.548, 'bval': 17.020 },
        { '#lines':  6, 'minval':  0.0, 'qthr':   0.054, 'norm': 0.536, 'bval': 17.374 },
        { '#lines':  6, 'minval':  0.0, 'qthr':   0.054, 'norm': 0.550, 'bval': 17.744 },
        { '#lines':  7, 'minval':  0.0, 'qthr':   0.063, 'norm': 0.532, 'bval': 18.104 },
        { '#lines':  7, 'minval':  0.0, 'qthr':   0.063, 'norm': 0.504, 'bval': 18.447 },
        { '#lines':  7, 'minval':  0.0, 'qthr':   0.063, 'norm': 0.496, 'bval': 18.782 },
        { '#lines':  9, 'minval':  0.0, 'qthr':   0.081, 'norm': 0.517, 'bval': 19.130 },
        { '#lines':  9, 'minval':  0.0, 'qthr':   0.081, 'norm': 0.527, 'bval': 19.487 },
        { '#lines':  9, 'minval':  0.0, 'qthr':   0.081, 'norm': 0.516, 'bval': 19.838 },
        { '#lines': 10, 'minval':  0.0, 'qthr':   0.180, 'norm': 0.497, 'bval': 20.179 },
        { '#lines': 10, 'minval':  0.0, 'qthr':   0.180, 'norm': 0.489, 'bval': 20.510 },
        { '#lines': 11, 'minval':  0.0, 'qthr':   0.198, 'norm': 0.502, 'bval': 20.852 },
        { '#lines': 14, 'minval':  0.0, 'qthr':   0.400, 'norm': 0.501, 'bval': 21.196 },
        { '#lines': 14, 'minval':  0.0, 'qthr':   0.400, 'norm': 0.491, 'bval': 21.531 },
        { '#lines': 15, 'minval':  0.0, 'qthr':   0.429, 'norm': 0.497, 'bval': 21.870 },
        { '#lines': 20, 'minval':  0.0, 'qthr':   1.808, 'norm': 0.504, 'bval': 22.214 },
        { '#lines': 20, 'minval':  0.0, 'qthr':   1.808, 'norm': 0.504, 'bval': 22.558 },
        { '#lines': 21, 'minval':  0.0, 'qthr':   1.898, 'norm': 0.495, 'bval': 22.898 },
        { '#lines': 27, 'minval':  0.0, 'qthr':  24.416, 'norm': 0.486, 'bval': 23.232 },
        { '#lines': 27, 'minval':  0.0, 'qthr':  24.416, 'norm': 0.484, 'bval': 23.564 },
        { '#lines': 36, 'minval':  0.0, 'qthr':  32.554, 'norm': 0.483, 'bval': 23.897 },
        { '#lines': 73, 'minval':  0.0, 'qthr': 660.124, 'norm': 0.475, 'bval': 24.229 },
        { '#lines': 18, 'minval':  0.0, 'qthr': 162.770, 'norm': 0.515, 'bval': 24.442 },
    ],
    'long32000': [
        { '#lines':  2, 'minval': 24.5, 'qthr':  4.532, 'norm': 0.997, 'bval':  0.313 },
        { '#lines':  2, 'minval': 24.5, 'qthr':  4.532, 'norm': 0.893, 'bval':  0.938 },
        { '#lines':  2, 'minval': 24.5, 'qthr':  1.809, 'norm': 0.881, 'bval':  1.563 },
        { '#lines':  2, 'minval': 20.0, 'qthr':  0.181, 'norm': 0.873, 'bval':  2.188 },
        { '#lines':  2, 'minval': 20.0, 'qthr':  0.181, 'norm': 0.872, 'bval':  2.813 },
        { '#lines':  2, 'minval': 20.0, 'qthr':  0.057, 'norm': 0.871, 'bval':  3.438 },
        { '#lines':  2, 'minval': 20.0, 'qthr':  0.018, 'norm': 0.860, 'bval':  4.045 },
        { '#lines':  2, 'minval': 20.0, 'qthr':  0.018, 'norm': 0.839, 'bval':  4.625 },
        { '#lines':  2, 'minval': 18.0, 'qthr':  0.018, 'norm': 0.812, 'bval':  5.173 },
        { '#lines':  2, 'minval': 18.0, 'qthr':  0.018, 'norm': 0.784, 'bval':  5.698 },
        { '#lines':  2, 'minval': 12.0, 'qthr':  0.018, 'norm': 0.741, 'bval':  6.185 },
        { '#lines':  2, 'minval': 12.0, 'qthr':  0.018, 'norm': 0.697, 'bval':  6.634 },
        { '#lines':  2, 'minval':  6.0, 'qthr':  0.018, 'norm': 0.674, 'bval':  7.070 },
        { '#lines':  2, 'minval':  6.0, 'qthr':  0.018, 'norm': 0.651, 'bval':  7.492 },
        { '#lines':  2, 'minval':  6.0, 'qthr':  0.018, 'norm': 0.633, 'bval':  7.905 },
        { '#lines':  2, 'minval':  3.0, 'qthr':  0.018, 'norm': 0.611, 'bval':  8.305 },
        { '#lines':  2, 'minval':  3.0, 'qthr':  0.018, 'norm': 0.589, 'bval':  8.695 },
        { '#lines':  2, 'minval':  3.0, 'qthr':  0.018, 'norm': 0.575, 'bval':  9.064 },
        { '#lines':  3, 'minval':  3.0, 'qthr':  0.027, 'norm': 0.654, 'bval':  9.484 },
        { '#lines':  3, 'minval':  3.0, 'qthr':  0.027, 'norm': 0.724, 'bval':  9.966 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.701, 'bval': 10.426 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.673, 'bval': 10.866 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.631, 'bval': 11.279 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.592, 'bval': 11.669 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.553, 'bval': 12.042 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.510, 'bval': 12.386 },
        { '#lines':  3, 'minval':  0.0, 'qthr':  0.027, 'norm': 0.506, 'bval': 12.721 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.562, 'bval': 13.091 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.598, 'bval': 13.488 },
        { '#lines':  4, 'minval':  0.0, 'qthr':  0.036, 'norm': 0.589, 'bval': 13.873 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.607, 'bval': 14.268 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.620, 'bval': 14.679 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.580, 'bval': 15.067 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.532, 'bval': 15.424 },
        { '#lines':  5, 'minval':  0.0, 'qthr':  0.045, 'norm': 0.517, 'bval': 15.771 },
        { '#lines':  6, 'minval':  0.0, 'qthr':  0.054, 'norm': 0.517, 'bval': 16.120 },
        { '#lines':  6, 'minval':  0.0, 'qthr':  0.054, 'norm': 0.509, 'bval': 16.466 },
        { '#lines':  6, 'minval':  0.0, 'qthr':  0.054, 'norm': 0.506, 'bval': 16.807 },
        { '#lines':  8, 'minval':  0.0, 'qthr':  0.072, 'norm': 0.522, 'bval': 17.158 },
        { '#lines':  8, 'minval':  0.0, 'qthr':  0.072, 'norm': 0.531, 'bval': 17.518 },
        { '#lines':  8, 'minval':  0.0, 'qthr':  0.072, 'norm': 0.519, 'bval': 17.869 },
        { '#lines': 10, 'minval':  0.0, 'qthr':  0.090, 'norm': 0.512, 'bval': 18.215 },
        { '#lines': 10, 'minval':  0.0, 'qthr':  0.090, 'norm': 0.509, 'bval': 18.563 },
        { '#lines': 10, 'minval':  0.0, 'qthr':  0.090, 'norm': 0.498, 'bval': 18.902 },
        { '#lines': 12, 'minval':  0.0, 'qthr':  0.109, 'norm': 0.494, 'bval': 19.239 },
        { '#lines': 12, 'minval':  0.0, 'qthr':  0.109, 'norm': 0.501, 'bval': 19.580 },
        { '#lines': 13, 'minval':  0.0, 'qthr':  0.118, 'norm': 0.508, 'bval': 19.925 },
        { '#lines': 14, 'minval':  0.0, 'qthr':  0.252, 'norm': 0.502, 'bval': 20.269 },
        { '#lines': 14, 'minval':  0.0, 'qthr':  0.252, 'norm': 0.493, 'bval': 20.606 },
        { '#lines': 16, 'minval':  0.0, 'qthr':  0.288, 'norm': 0.497, 'bval': 20.944 },
        { '#lines': 20, 'minval':  0.0, 'qthr':  0.572, 'norm': 0.506, 'bval': 21.288 },
        { '#lines': 20, 'minval':  0.0, 'qthr':  0.572, 'norm': 0.510, 'bval': 21.635 },
        { '#lines': 23, 'minval':  0.0, 'qthr':  0.658, 'norm': 0.504, 'bval': 21.980 },
        { '#lines': 27, 'minval':  0.0, 'qthr':  2.441, 'norm': 0.496, 'bval': 22.319 },
        { '#lines': 27, 'minval':  0.0, 'qthr':  2.441, 'norm': 0.493, 'bval': 22.656 },
        { '#lines': 32, 'minval':  0.0, 'qthr':  2.893, 'norm': 0.490, 'bval': 22.993 },
        { '#lines': 37, 'minval':  0.0, 'qthr': 33.458, 'norm': 0.482, 'bval': 23.326 },
        { '#lines': 37, 'minval':  0.0, 'qthr': 33.458, 'norm': 0.458, 'bval': 23.656 },
        { '#lines': 12, 'minval':  0.0, 'qthr': 10.851, 'norm': 0.500, 'bval': 23.937 },
    ],
    'short48000': [
        { '#lines':  1, 'qthr':   4.532, 'norm': 1.000, 'SNR': -8.240, 'bval':  0.000 },
        { '#lines':  1, 'qthr':   0.904, 'norm': 0.989, 'SNR': -8.240, 'bval':  1.875 },
        { '#lines':  1, 'qthr':   0.029, 'norm': 0.989, 'SNR': -8.240, 'bval':  3.750 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.981, 'SNR': -8.240, 'bval':  5.438 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.985, 'SNR': -8.240, 'bval':  6.857 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.984, 'SNR': -8.240, 'bval':  8.109 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.980, 'SNR': -8.240, 'bval':  9.237 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.968, 'SNR': -8.240, 'bval': 10.202 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.954, 'SNR': -8.240, 'bval': 11.083 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.929, 'SNR': -8.240, 'bval': 11.865 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.906, 'SNR': -7.447, 'bval': 12.554 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.883, 'SNR': -7.447, 'bval': 13.195 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.844, 'SNR': -7.447, 'bval': 13.781 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.792, 'SNR': -7.447, 'bval': 14.309 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.747, 'SNR': -7.447, 'bval': 14.803 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.689, 'SNR': -7.447, 'bval': 15.250 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.644, 'SNR': -7.447, 'bval': 15.667 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.592, 'SNR': -7.447, 'bval': 16.068 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.553, 'SNR': -7.447, 'bval': 16.409 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.850, 'SNR': -7.447, 'bval': 17.045 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.811, 'SNR': -6.990, 'bval': 17.607 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.736, 'SNR': -6.990, 'bval': 18.097 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.665, 'SNR': -6.990, 'bval': 18.528 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.610, 'SNR': -6.990, 'bval': 18.931 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.544, 'SNR': -6.990, 'bval': 19.295 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.528, 'SNR': -6.990, 'bval': 19.636 },
        { '#lines':  3, 'qthr':   0.054, 'norm': 0.621, 'SNR': -6.990, 'bval': 20.038 },
        { '#lines':  3, 'qthr':   0.054, 'norm': 0.673, 'SNR': -6.990, 'bval': 20.486 },
        { '#lines':  3, 'qthr':   0.054, 'norm': 0.635, 'SNR': -6.990, 'bval': 20.900 },
        { '#lines':  4, 'qthr':   0.114, 'norm': 0.626, 'SNR': -6.990, 'bval': 21.306 },
        { '#lines':  4, 'qthr':   0.114, 'norm': 0.636, 'SNR': -6.020, 'bval': 21.722 },
        { '#lines':  5, 'qthr':   0.452, 'norm': 0.615, 'SNR': -6.020, 'bval': 22.128 },
        { '#lines':  5, 'qthr':   0.452, 'norm': 0.579, 'SNR': -6.020, 'bval': 22.513 },
        { '#lines':  5, 'qthr':   0.452, 'norm': 0.551, 'SNR': -6.020, 'bval': 22.877 },
        { '#lines':  7, 'qthr':   6.330, 'norm': 0.552, 'SNR': -5.229, 'bval': 23.241 },
        { '#lines':  7, 'qthr':   6.330, 'norm': 0.559, 'SNR': -5.229, 'bval': 23.616 },
        { '#lines': 11, 'qthr':   9.947, 'norm': 0.528, 'SNR': -5.229, 'bval': 23.974 },
        { '#lines': 17, 'qthr': 153.727, 'norm': 0.479, 'SNR': -5.229, 'bval': 24.313 },
    ],
    'short44100': [
        { '#lines':  1, 'qthr':   4.532, 'norm': 1.000, 'SNR': -8.240, 'bval':  0.000 },
        { '#lines':  1, 'qthr':   0.904, 'norm': 0.983, 'SNR': -8.240, 'bval':  1.723 },
        { '#lines':  1, 'qthr':   0.029, 'norm': 0.983, 'SNR': -8.240, 'bval':  3.445 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.982, 'SNR': -8.240, 'bval':  5.057 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.985, 'SNR': -8.240, 'bval':  6.422 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.983, 'SNR': -8.240, 'bval':  7.609 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.978, 'SNR': -8.240, 'bval':  8.710 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.967, 'SNR': -8.240, 'bval':  9.662 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.948, 'SNR': -8.240, 'bval': 10.515 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.930, 'SNR': -8.240, 'bval': 11.293 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.914, 'SNR': -7.447, 'bval': 12.009 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.870, 'SNR': -7.447, 'bval': 12.625 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.845, 'SNR': -7.447, 'bval': 13.210 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.800, 'SNR': -7.447, 'bval': 13.748 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.749, 'SNR': -7.447, 'bval': 14.241 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.701, 'SNR': -7.447, 'bval': 14.695 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.653, 'SNR': -7.447, 'bval': 15.125 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.590, 'SNR': -7.447, 'bval': 15.508 },
        { '#lines':  1, 'qthr':   0.009, 'norm': 0.616, 'SNR': -7.447, 'bval': 15.891 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.860, 'SNR': -7.447, 'bval': 16.537 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.823, 'SNR': -6.990, 'bval': 17.112 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.762, 'SNR': -6.990, 'bval': 17.621 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.688, 'SNR': -6.990, 'bval': 18.073 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.612, 'SNR': -6.990, 'bval': 18.470 },
        { '#lines':  2, 'qthr':   0.018, 'norm': 0.594, 'SNR': -6.990, 'bval': 18.849 },
        { '#lines':  3, 'qthr':   0.027, 'norm': 0.658, 'SNR': -6.990, 'bval': 19.271 },
        { '#lines':  3, 'qthr':   0.027, 'norm': 0.706, 'SNR': -6.990, 'bval': 19.741 },
        { '#lines':  3, 'qthr':   0.054, 'norm': 0.660, 'SNR': -6.990, 'bval': 20.177 },
        { '#lines':  3, 'qthr':   0.054, 'norm': 0.606, 'SNR': -6.990, 'bval': 20.576 },
        { '#lines':  3, 'qthr':   0.054, 'norm': 0.565, 'SNR': -6.990, 'bval': 20.950 },
        { '#lines':  4, 'qthr':   0.114, 'norm': 0.560, 'SNR': -6.020, 'bval': 21.316 },
        { '#lines':  4, 'qthr':   0.114, 'norm': 0.579, 'SNR': -6.020, 'bval': 21.699 },
        { '#lines':  5, 'qthr':   0.452, 'norm': 0.567, 'SNR': -6.020, 'bval': 22.078 },
        { '#lines':  5, 'qthr':   0.452, 'norm': 0.534, 'SNR': -6.020, 'bval': 22.438 },
        { '#lines':  5, 'qthr':   0.452, 'norm': 0.514, 'SNR': -5.229, 'bval': 22.782 },
        { '#lines':  7, 'qthr':   6.330, 'norm': 0.520, 'SNR': -5.229, 'bval': 23.133 },
        { '#lines':  7, 'qthr':   6.330, 'norm': 0.518, 'SNR': -5.229, 'bval': 23.484 },
        { '#lines':  7, 'qthr':   6.330, 'norm': 0.507, 'SNR': -5.229, 'bval': 23.828 },
        { '#lines': 19, 'qthr': 171.813, 'norm': 0.447, 'SNR': -4.559, 'bval': 24.173 },
    ],
    'short32000': [
        { '#lines':  1, 'qthr': 4.532, 'norm': 1.000, 'SNR': -8.240, 'bval':  0.000 },
        { '#lines':  1, 'qthr': 0.904, 'norm': 0.985, 'SNR': -8.240, 'bval':  1.250 },
        { '#lines':  1, 'qthr': 0.090, 'norm': 0.983, 'SNR': -8.240, 'bval':  2.500 },
        { '#lines':  1, 'qthr': 0.029, 'norm': 0.983, 'SNR': -8.240, 'bval':  3.750 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.981, 'SNR': -8.240, 'bval':  4.909 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.975, 'SNR': -8.240, 'bval':  5.958 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.959, 'SNR': -8.240, 'bval':  6.857 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.944, 'SNR': -8.240, 'bval':  7.700 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.933, 'SNR': -8.240, 'bval':  8.500 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.920, 'SNR': -8.240, 'bval':  9.237 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.892, 'SNR': -7.447, 'bval':  9.895 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.863, 'SNR': -7.447, 'bval': 10.500 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.839, 'SNR': -7.447, 'bval': 11.083 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.786, 'SNR': -7.447, 'bval': 11.604 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.755, 'SNR': -7.447, 'bval': 12.107 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.698, 'SNR': -7.447, 'bval': 12.554 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.673, 'SNR': -7.447, 'bval': 13.000 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.605, 'SNR': -7.447, 'bval': 13.391 },
        { '#lines':  1, 'qthr': 0.009, 'norm': 0.629, 'SNR': -7.447, 'bval': 13.781 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.883, 'SNR': -7.447, 'bval': 14.474 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.858, 'SNR': -6.990, 'bval': 15.096 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.829, 'SNR': -6.990, 'bval': 15.667 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.767, 'SNR': -6.990, 'bval': 16.177 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.705, 'SNR': -6.990, 'bval': 16.636 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.637, 'SNR': -6.990, 'bval': 17.057 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.564, 'SNR': -6.990, 'bval': 17.429 },
        { '#lines':  2, 'qthr': 0.018, 'norm': 0.550, 'SNR': -6.990, 'bval': 17.786 },
        { '#lines':  3, 'qthr': 0.027, 'norm': 0.603, 'SNR': -6.990, 'bval': 18.177 },
        { '#lines':  3, 'qthr': 0.027, 'norm': 0.635, 'SNR': -6.990, 'bval': 18.597 },
        { '#lines':  3, 'qthr': 0.027, 'norm': 0.592, 'SNR': -6.990, 'bval': 18.994 },
        { '#lines':  3, 'qthr': 0.027, 'norm': 0.533, 'SNR': -6.020, 'bval': 19.352 },
        { '#lines':  3, 'qthr': 0.027, 'norm': 0.518, 'SNR': -6.020, 'bval': 19.693 },
        { '#lines':  4, 'qthr': 0.072, 'norm': 0.568, 'SNR': -6.020, 'bval': 20.066 },
        { '#lines':  4, 'qthr': 0.072, 'norm': 0.594, 'SNR': -6.020, 'bval': 20.462 },
        { '#lines':  4, 'qthr': 0.072, 'norm': 0.568, 'SNR': -5.229, 'bval': 20.841 },
        { '#lines':  5, 'qthr': 0.143, 'norm': 0.536, 'SNR': -5.229, 'bval': 21.201 },
        { '#lines':  5, 'qthr': 0.143, 'norm': 0.522, 'SNR': -5.229, 'bval': 21.549 },
        { '#lines':  6, 'qthr': 0.172, 'norm': 0.542, 'SNR': -5.229, 'bval': 21.911 },
        { '#lines':  7, 'qthr': 0.633, 'norm': 0.539, 'SNR': -4.559, 'bval': 22.275 },
        { '#lines':  7, 'qthr': 0.633, 'norm': 0.519, 'SNR': -4.559, 'bval': 22.625 },
        { '#lines':  8, 'qthr': 0.723, 'norm': 0.514, 'SNR': -3.980, 'bval': 22.971 },
        { '#lines': 10, 'qthr': 9.043, 'norm': 0.518, 'SNR': -3.980, 'bval': 23.321 },
    ],
}

PSYCO_DATA = {
    'long48000': [
        { 'cbw': 3, 'bu':  0, 'bo': 4, 'w1': 1.000, 'w2': 0.056 },
        { 'cbw': 3, 'bu':  4, 'bo': 7, 'w1': 0.944, 'w2': 0.611 },
        { 'cbw': 4, 'bu':  7, 'bo':11, 'w1': 0.389, 'w2': 0.167 },
        { 'cbw': 3, 'bu': 11, 'bo':14, 'w1': 0.833, 'w2': 0.722 },
        { 'cbw': 3, 'bu': 14, 'bo':17, 'w1': 0.278, 'w2': 0.639 },
        { 'cbw': 2, 'bu': 17, 'bo':19, 'w1': 0.361, 'w2': 0.417 },
        { 'cbw': 3, 'bu': 19, 'bo':22, 'w1': 0.583, 'w2': 0.083 },
        { 'cbw': 2, 'bu': 22, 'bo':24, 'w1': 0.917, 'w2': 0.750 },
        { 'cbw': 3, 'bu': 24, 'bo':27, 'w1': 0.250, 'w2': 0.417 },
        { 'cbw': 3, 'bu': 27, 'bo':30, 'w1': 0.583, 'w2': 0.648 },
        { 'cbw': 3, 'bu': 30, 'bo':33, 'w1': 0.352, 'w2': 0.611 },
        { 'cbw': 3, 'bu': 33, 'bo':36, 'w1': 0.389, 'w2': 0.625 },
        { 'cbw': 4, 'bu': 36, 'bo':40, 'w1': 0.375, 'w2': 0.144 },
        { 'cbw': 3, 'bu': 40, 'bo':43, 'w1': 0.856, 'w2': 0.389 },
        { 'cbw': 3, 'bu': 43, 'bo':46, 'w1': 0.611, 'w2': 0.160 },
        { 'cbw': 3, 'bu': 46, 'bo':49, 'w1': 0.840, 'w2': 0.217 },
        { 'cbw': 3, 'bu': 49, 'bo':52, 'w1': 0.783, 'w2': 0.184 },
        { 'cbw': 2, 'bu': 52, 'bo':54, 'w1': 0.816, 'w2': 0.886 },
        { 'cbw': 3, 'bu': 54, 'bo':57, 'w1': 0.114, 'w2': 0.313 },
        { 'cbw': 2, 'bu': 57, 'bo':59, 'w1': 0.687, 'w2': 0.452 },
        { 'cbw': 1, 'bu': 59, 'bo':60, 'w1': 0.548, 'w2': 0.908 },
    ],
    'long44100': [
        { 'cbw': 3, 'bu':  0, 'bo': 4, 'w1': 1.000, 'w2': 0.056 },
        { 'cbw': 3, 'bu':  4, 'bo': 7, 'w1': 0.944, 'w2': 0.611 },
        { 'cbw': 4, 'bu':  7, 'bo':11, 'w1': 0.389, 'w2': 0.167 },
        { 'cbw': 3, 'bu': 11, 'bo':14, 'w1': 0.833, 'w2': 0.722 },
        { 'cbw': 3, 'bu': 14, 'bo':17, 'w1': 0.278, 'w2': 0.139 },
        { 'cbw': 1, 'bu': 17, 'bo':18, 'w1': 0.861, 'w2': 0.917 },
        { 'cbw': 3, 'bu': 18, 'bo':21, 'w1': 0.083, 'w2': 0.583 },
        { 'cbw': 3, 'bu': 21, 'bo':24, 'w1': 0.417, 'w2': 0.250 },
        { 'cbw': 3, 'bu': 24, 'bo':27, 'w1': 0.750, 'w2': 0.805 },
        { 'cbw': 3, 'bu': 27, 'bo':30, 'w1': 0.194, 'w2': 0.574 },
        { 'cbw': 3, 'bu': 30, 'bo':33, 'w1': 0.426, 'w2': 0.537 },
        { 'cbw': 3, 'bu': 33, 'bo':36, 'w1': 0.463, 'w2': 0.819 },
        { 'cbw': 4, 'bu': 36, 'bo':40, 'w1': 0.180, 'w2': 0.100 },
        { 'cbw': 3, 'bu': 40, 'bo':43, 'w1': 0.900, 'w2': 0.468 },
        { 'cbw': 3, 'bu': 43, 'bo':46, 'w1': 0.532, 'w2': 0.623 },
        { 'cbw': 3, 'bu': 46, 'bo':49, 'w1': 0.376, 'w2': 0.450 },
        { 'cbw': 3, 'bu': 49, 'bo':52, 'w1': 0.550, 'w2': 0.552 },
        { 'cbw': 3, 'bu': 52, 'bo':55, 'w1': 0.448, 'w2': 0.403 },
        { 'cbw': 2, 'bu': 55, 'bo':57, 'w1': 0.597, 'w2': 0.643 },
        { 'cbw': 2, 'bu': 57, 'bo':59, 'w1': 0.357, 'w2': 0.722 },
        { 'cbw': 2, 'bu': 59, 'bo':61, 'w1': 0.278, 'w2': 0.960 },
    ],
    'long32000': [
        { 'cbw': 1, 'bu':  0, 'bo': 2, 'w1': 1.000, 'w2': 0.528 },
        { 'cbw': 2, 'bu':  2, 'bo': 4, 'w1': 0.472, 'w2': 0.305 },
        { 'cbw': 2, 'bu':  4, 'bo': 6, 'w1': 0.694, 'w2': 0.083 },
        { 'cbw': 1, 'bu':  6, 'bo': 7, 'w1': 0.917, 'w2': 0.861 },
        { 'cbw': 2, 'bu':  7, 'bo': 9, 'w1': 0.139, 'w2': 0.639 },
        { 'cbw': 2, 'bu':  9, 'bo':11, 'w1': 0.361, 'w2': 0.417 },
        { 'cbw': 3, 'bu': 11, 'bo':14, 'w1': 0.583, 'w2': 0.083 },
        { 'cbw': 2, 'bu': 14, 'bo':16, 'w1': 0.917, 'w2': 0.750 },
        { 'cbw': 3, 'bu': 16, 'bo':19, 'w1': 0.250, 'w2': 0.870 },
        { 'cbw': 3, 'bu': 19, 'bo':22, 'w1': 0.130, 'w2': 0.833 },
        { 'cbw': 4, 'bu': 22, 'bo':26, 'w1': 0.167, 'w2': 0.389 },
        { 'cbw': 4, 'bu': 26, 'bo':30, 'w1': 0.611, 'w2': 0.478 },
        { 'cbw': 4, 'bu': 30, 'bo':34, 'w1': 0.522, 'w2': 0.033 },
        { 'cbw': 3, 'bu': 34, 'bo':37, 'w1': 0.967, 'w2': 0.917 },
        { 'cbw': 4, 'bu': 37, 'bo':41, 'w1': 0.083, 'w2': 0.617 },
        { 'cbw': 3, 'bu': 41, 'bo':44, 'w1': 0.383, 'w2': 0.995 },
        { 'cbw': 4, 'bu': 44, 'bo':48, 'w1': 0.005, 'w2': 0.274 },
        { 'cbw': 3, 'bu': 48, 'bo':51, 'w1': 0.726, 'w2': 0.480 },
        { 'cbw': 3, 'bu': 51, 'bo':54, 'w1': 0.519, 'w2': 0.261 },
        { 'cbw': 2, 'bu': 54, 'bo':56, 'w1': 0.739, 'w2': 0.884 },
        { 'cbw': 2, 'bu': 56, 'bo':58, 'w1': 0.116, 'w2': 1.000 },
    ],
    'short48000': [
        { 'cbw': 2, 'bu':  0, 'bo': 2, 'w1': 1.000, 'w2': 0.167 },
        { 'cbw': 2, 'bu':  3, 'bo': 5, 'w1': 0.833, 'w2': 0.833 },
        { 'cbw': 3, 'bu':  5, 'bo': 8, 'w1': 0.167, 'w2': 0.500 },
        { 'cbw': 3, 'bu':  8, 'bo':11, 'w1': 0.500, 'w2': 0.167 },
        { 'cbw': 4, 'bu': 11, 'bo':15, 'w1': 0.833, 'w2': 0.167 },
        { 'cbw': 4, 'bu': 15, 'bo':19, 'w1': 0.833, 'w2': 0.583 },
        { 'cbw': 3, 'bu': 19, 'bo':22, 'w1': 0.417, 'w2': 0.917 },
        { 'cbw': 4, 'bu': 22, 'bo':26, 'w1': 0.083, 'w2': 0.944 },
        { 'cbw': 4, 'bu': 26, 'bo':30, 'w1': 0.055, 'w2': 0.042 },
        { 'cbw': 2, 'bu': 30, 'bo':32, 'w1': 0.958, 'w2': 0.567 },
        { 'cbw': 3, 'bu': 32, 'bo':35, 'w1': 0.433, 'w2': 0.167 },
        { 'cbw': 2, 'bu': 35, 'bo':37, 'w1': 0.833, 'w2': 0.618 },
    ],
    'short44100': [
        { 'cbw': 2, 'bu':  0, 'bo': 2, 'w1': 1.000, 'w2': 0.167 },
        { 'cbw': 2, 'bu':  3, 'bo': 5, 'w1': 0.833, 'w2': 0.833 },
        { 'cbw': 3, 'bu':  5, 'bo': 8, 'w1': 0.167, 'w2': 0.500 },
        { 'cbw': 3, 'bu':  8, 'bo':11, 'w1': 0.500, 'w2': 0.167 },
        { 'cbw': 4, 'bu': 11, 'bo':15, 'w1': 0.833, 'w2': 0.167 },
        { 'cbw': 5, 'bu': 15, 'bo':20, 'w1': 0.833, 'w2': 0.250 },
        { 'cbw': 3, 'bu': 20, 'bo':23, 'w1': 0.750, 'w2': 0.583 },
        { 'cbw': 4, 'bu': 23, 'bo':27, 'w1': 0.417, 'w2': 0.055 },
        { 'cbw': 3, 'bu': 27, 'bo':30, 'w1': 0.944, 'w2': 0.375 },
        { 'cbw': 3, 'bu': 30, 'bo':33, 'w1': 0.625, 'w2': 0.300 },
        { 'cbw': 3, 'bu': 33, 'bo':36, 'w1': 0.700, 'w2': 0.167 },
        { 'cbw': 2, 'bu': 36, 'bo':38, 'w1': 0.833, 'w2': 1.000 },
    ],
    'short32000': [
        { 'cbw': 2, 'bu':  0, 'bo': 2, 'w1': 1.000, 'w2': 0.167 },
        { 'cbw': 2, 'bu':  3, 'bo': 5, 'w1': 0.833, 'w2': 0.833 },
        { 'cbw': 3, 'bu':  5, 'bo': 8, 'w1': 0.167, 'w2': 0.500 },
        { 'cbw': 3, 'bu':  8, 'bo':11, 'w1': 0.500, 'w2': 0.167 },
        { 'cbw': 4, 'bu': 11, 'bo':15, 'w1': 0.833, 'w2': 0.167 },
        { 'cbw': 5, 'bu': 15, 'bo':20, 'w1': 0.833, 'w2': 0.250 },
        { 'cbw': 4, 'bu': 20, 'bo':24, 'w1': 0.750, 'w2': 0.250 },
        { 'cbw': 5, 'bu': 24, 'bo':29, 'w1': 0.750, 'w2': 0.055 },
        { 'cbw': 4, 'bu': 29, 'bo':33, 'w1': 0.944, 'w2': 0.375 },
        { 'cbw': 4, 'bu': 33, 'bo':37, 'w1': 0.625, 'w2': 0.472 },
        { 'cbw': 3, 'bu': 37, 'bo':40, 'w1': 0.528, 'w2': 0.937 },
        { 'cbw': 1, 'bu': 40, 'bo':41, 'w1': 0.062, 'w2': 1.000 },
    ]
}

def get_frame(data, gr):
    '''
    グラニュールインデックスに対応する1024サンプルフレームの取得
    '''
    frame = np.zeros(1024)
    for i in range(1024):
        index = 576 * gr - 768 + i
        if index >= 0:
            frame[i] = data[index]
    return frame

def dist10fft(data):
    MIN_ABS = 0.0005 ** 0.5
    spec = np.fft.fft(data)
    spec = np.where(np.abs(spec) <= np.finfo(np.float64).min, 0.0, spec)
    spec = np.where(np.abs(spec) <= MIN_ABS, MIN_ABS, spec)
    return spec

def compute_spreading_function(partition):
    '''
    Spreading functionの計算
    '''
    part_max = len(partition)
    sfunc = np.zeros((part_max, part_max))
    for i in range(part_max):
        for j in range(part_max):
            tx = partition[i]['bval'] - partition[j]['bval']
            tx = 3.0 * tx if i <= j else 1.5 * tx
            xij = 8.0 * min((tx - 0.5) ** 2.0 - 2.0 * (tx - 0.5), 0.0)
            ty = 15.811389 + 7.5 * (tx + 0.474) - 17.5 * (1.0 + (tx + 0.474) ** 2.0) ** 0.5
            sfunc[i][j] = 10.0 ** ((xij + ty) / 10) if ty >= -60 else 0.0 # 本とdist10で異なる
    return sfunc

def compute_unpredictability(wl, ws, prev_wl, prevprev_wl):
    '''
    Unpredictability cwの計算
    '''
    cw = np.zeros(513)
    # 振幅・位相の直線予測結果
    wlprimeabs = 2.0 * np.abs(prev_wl) - np.abs(prevprev_wl)
    wlprimearg = 2.0 * np.angle(prev_wl) - np.angle(prevprev_wl)
    wlprime = wlprimeabs * np.exp(1j * wlprimearg)
    # 直線予測との差
    diffwl = wl - wlprime
    for j in range(6):
        numer = np.abs(wl[j]) + np.abs(wlprime[j])
        if numer > 0.0:
            cw[j] = np.abs(diffwl[j]) / numer
    predwsabs = 2.0 * np.abs(ws[0]) - np.abs(ws[2])
    predwsarg = 2.0 * np.angle(ws[0]) - np.angle(ws[2])
    predws = predwsabs * np.exp(1j * predwsarg)
    diffws = ws[1] - predws
    for j in np.arange(6, 206, 4):
        k = (j + 2) // 4
        numer = np.abs(ws[1][k]) + np.abs(predws[k])
        if numer > 0.0:
            cw[j] = np.abs(diffws[k]) / numer
            cw[j + 1] = cw[j + 2] = cw[j + 3] = cw[j]
    # 残りは0.4で埋める
    for j in np.arange(206, len(cw)):
        cw[j] = 0.4

    return cw

if __name__ == '__main__':
    # print(PARTITION_DATA)
    # print(PSYCO_DATA)

    NUM_CRITICAL_BANDS = 63
    NUM_CRITICAL_BANDS_SHORT = 42
    PERCETUAL_ENTROPY_THRESHOLD = 1800.0

    sf, data = wavfile.read('repeat.wav')
    NUM_CHANNELS = data.shape[1]

    LONG_WINDOW = [0.5 * (1.0 - np.cos(2.0 * np.pi * (i - 0.5) / 1024)) for i in range(1024)]
    SHORT_WINDOW =[0.5 * (1.0 - np.cos(2.0 * np.pi * (i - 0.5) / 256)) for i in range(256)]

    PARTITION_LONG = PARTITION_DATA[f'long{sf}']
    PARTITION_SHORT = PARTITION_DATA[f'short{sf}']
    PSYCO_LONG = PSYCO_DATA[f'long{sf}']
    PSYCO_SHORT = PSYCO_DATA[f'short{sf}']

    # 分割インデックス作成
    PARTITION_LONG_INDEX = np.zeros(513, dtype=int)
    index = 0
    for part_index, part in enumerate(PARTITION_LONG):
        for _ in range(part['#lines']):
            PARTITION_LONG_INDEX[index] = part_index
            index += 1
    PARTITION_SHORT_INDEX = np.zeros(129, dtype=int)
    index = 0
    for part_index, part in enumerate(PARTITION_SHORT):
        for _ in range(part['#lines']):
            PARTITION_SHORT_INDEX[index] = part_index
            index += 1

    prev_wl = np.zeros((NUM_CHANNELS, 1024), dtype=complex)
    prevprev_wl = np.zeros((NUM_CHANNELS, 1024), dtype=complex)

    prev_nb = np.zeros((NUM_CHANNELS, NUM_CRITICAL_BANDS))
    prevprev_nb = np.zeros((NUM_CHANNELS, NUM_CRITICAL_BANDS))

    ratio_long = np.zeros((NUM_CHANNELS, len(PSYCO_LONG)))
    ratio_short = np.zeros((NUM_CHANNELS, len(PSYCO_SHORT), 3))

    prev_block_type = [ 'NORMAL', 'NORMAL' ]

    SPREADING_FUNCTION_LONG = compute_spreading_function(PARTITION_LONG)
    SPREADING_FUNCTION_SHORT = compute_spreading_function(PARTITION_SHORT)

    for gr in range(5):
        for ch in range(NUM_CHANNELS):
            # フレーム取得
            frame = get_frame(data.T[ch], gr)

            # FFT
            wl = dist10fft(frame * LONG_WINDOW)
            ws = []
            for b in range(3):
                short_frame = frame[128 * b + 256 + np.arange(256)].copy()
                ws.append(dist10fft(short_frame * SHORT_WINDOW))

            # Unpredictability計算
            cw = compute_unpredictability(wl, ws, prev_wl[ch], prevprev_wl[ch])
            # 状態更新
            prevprev_wl[ch] = prev_wl[ch]
            prev_wl[ch] = wl

            # パーティションごとのエネルギー計算
            energy_long = np.abs(wl) ** 2
            energy_short = []
            for sblock in range(3):
                energy_short.append(np.abs(ws[sblock]) ** 2)
            eb = np.zeros(NUM_CRITICAL_BANDS)
            cb = np.zeros(NUM_CRITICAL_BANDS)
            for j in range(513):
                tp = PARTITION_LONG_INDEX[j]
                if tp >= 0:
                    # BUG?: PARTITION_LONG_INDEXの63以降で0となっておりtp==0が過剰に加算される dist10でも同様
                    eb[tp] += energy_long[j]
                    cb[tp] += cw[j] * energy_long[j]

            # 広がり関数(Spreading Function)と畳み込み
            ecb = np.zeros(NUM_CRITICAL_BANDS)
            ctb = np.zeros(NUM_CRITICAL_BANDS)
            for b in range(NUM_CRITICAL_BANDS):
                for k in range(NUM_CRITICAL_BANDS):
                    ecb[b] += SPREADING_FUNCTION_LONG[b][k] * eb[k]
                    ctb[b] += SPREADING_FUNCTION_LONG[b][k] * cb[k]
            
            # 信号対ノイズ比（SNR）の計算
            snr = np.zeros(NUM_CRITICAL_BANDS)
            nb = np.zeros(NUM_CRITICAL_BANDS)
            for b in range(NUM_CRITICAL_BANDS):
                cbb = 0.0
                if ecb[b] != 0.0:
                    cbb = np.log(max(ctb[b] / ecb[b], 0.01))
                tbb = min(1.0, max(0.0, - 0.299 - 0.43 * cbb))
                snr[b] = max(PARTITION_LONG[b]['minval'], 29.0 * tbb + 6.0 * (1.0 - tbb))
                nb[b] = PARTITION_LONG[b]['norm'] * ecb[b] * 10.0 ** (-snr[b] / 10.0)

            # 各パーティションの聴覚閾値を計算
            thr = np.zeros(NUM_CRITICAL_BANDS)
            for b in range(NUM_CRITICAL_BANDS):
                thr[b] = min(nb[b], min(2.0 * prev_nb[ch][b], 16.0 * prevprev_nb[ch][b]))
                thr[b] = max(thr[b], PARTITION_LONG[b]['qthr'])
            prevprev_nb[ch] = prev_nb[ch]
            prev_nb[ch] = nb

            # 知覚エントロピー(percetual entropy)の計算
            pe = 0.0
            for b in range(NUM_CRITICAL_BANDS):
                # BUG: numlinesの先頭部分がショートブロックの値になっている。
                # L3para_readのバグ。一度ロングブロックで読み込ませて、同一の領域にショートブロックの値を読み込ませている
                bug_lines = PARTITION_LONG[b]['#lines']
                if b < len(PARTITION_SHORT):
                    bug_lines = PARTITION_SHORT[b]['#lines']
                tp = min(0.0, np.log((thr[b] + 1.0) / (eb[b] + 1.0)))
                pe -= bug_lines * tp
                # 正しくは以下の計算式のはず
                # pe -= PARTITION_LONG[b]['#lines'] * tp

            # ブロックタイプ確定・スケールファクタバンドの聴覚閾値計算
            if pe < PERCETUAL_ENTROPY_THRESHOLD:
                if prev_block_type[ch] == 'NORMAL' or prev_block_type[ch] == 'STOP':
                    block_type = 'NORMAL'
                elif prev_block_type[ch] == 'SHORT':
                    block_type = 'STOP'
                else:
                    assert(0)
                for sb, psy in enumerate(PSYCO_LONG):
                    bu = psy['bu']
                    bo = psy['bo']
                    en = psy['w1'] * eb[bu] + psy['w2'] * eb[bo]
                    thm = psy['w1'] * thr[bu] + psy['w2'] * thr[bo]
                    for b in np.arange(bu + 1, bo):
                        en += eb[b]
                        thm += thr[b]
                    ratio_long[ch][sb] = thm / en if en != 0.0 else 0.0
            else:
                block_type = 'SHORT'
                if prev_block_type[ch] == 'NORMAL':
                    prev_block_type[ch] = 'START'
                elif prev_block_type[ch] == 'STOP':
                    prev_block_type[ch] = 'SHORT'
                snr_short = np.zeros(NUM_CRITICAL_BANDS_SHORT)
                qthr_short = np.zeros(NUM_CRITICAL_BANDS_SHORT)
                for b in range(len(PARTITION_SHORT)):
                    snr_short[b] = PARTITION_SHORT[b]['SNR']
                    qthr_short[b] = PARTITION_SHORT[b]['qthr']
                for sblock in range(3):
                    eb = np.zeros(NUM_CRITICAL_BANDS_SHORT)
                    ecb = np.zeros(NUM_CRITICAL_BANDS_SHORT)
                    for j in range(129):
                        eb[PARTITION_SHORT_INDEX[j]] += energy_short[sblock][j]
                    for b in range(NUM_CRITICAL_BANDS_SHORT):
                        for k in range(NUM_CRITICAL_BANDS_SHORT):
                            # longパーティション使ってるのバグでは？
                            ecb[b] += SPREADING_FUNCTION_LONG[b][k] * eb[k]
                    for b in range(NUM_CRITICAL_BANDS_SHORT):
                        # longパーティション使ってるのバグでは？
                        nb[b] = ecb[b] * PARTITION_LONG[b]['norm'] * 10.0 ** (snr_short[b] / 10.0)
                        thr[b] = max(qthr_short[b], nb[b])
                    for sb, psy in enumerate(PSYCO_SHORT):
                        bu = psy['bu']
                        bo = psy['bo']
                        en = psy['w1'] * eb[bu] + psy['w2'] * eb[bo]
                        thm = psy['w1'] * thr[bu] + psy['w2'] * thr[bo]
                        for b in np.arange(bu + 1, bo):
                            en += eb[b]
                            thm += thr[b]
                        ratio_short[ch][sb][sblock] = thm / en if en != 0.0 else 0.0
            prev_block_type[ch] = block_type
