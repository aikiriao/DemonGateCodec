import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

ENCODER_C_COEF = [
    0.000000000, -0.000000477, -0.000000477, -0.000000477,
    -0.000000477, -0.000000477, -0.000000477, -0.000000954,
    -0.000000954, -0.000000954, -0.000000954, -0.000001431,
    -0.000001431, -0.000001907, -0.000001907, -0.000002384,
    -0.000002384, -0.000002861, -0.000003338, -0.000003338,
    -0.000003815, -0.000004292, -0.000004768, -0.000005245,
    -0.000006199, -0.000006676, -0.000007629, -0.000008106,
    -0.000009060, -0.000010014, -0.000011444, -0.000012398,
    -0.000013828, -0.000014782, -0.000016689, -0.000018120,
    -0.000019550, -0.000021458, -0.000023365, -0.000025272,
    -0.000027657, -0.000030041, -0.000032425, -0.000034809,
    -0.000037670, -0.000040531, -0.000043392, -0.000046253,
    -0.000049591, -0.000052929, -0.000055790, -0.000059605,
    -0.000062943, -0.000066280, -0.000070095, -0.000073433,
    -0.000076771, -0.000080585, -0.000083923, -0.000087261,
    -0.000090599, -0.000093460, -0.000096321, -0.000099182,
    0.000101566,  0.000103951,  0.000105858,  0.000107288,
    0.000108242,  0.000108719,  0.000108719,  0.000108242,
    0.000106812,  0.000105381,  0.000102520,  0.000099182,
    0.000095367,  0.000090122,  0.000084400,  0.000077724,
    0.000069618,  0.000060558,  0.000050545,  0.000039577,
    0.000027180,  0.000013828, -0.000000954, -0.000017166,
    -0.000034332, -0.000052929, -0.000072956, -0.000093937,
    -0.000116348, -0.000140190, -0.000165462, -0.000191212,
    -0.000218868, -0.000247478, -0.000277042, -0.000307560,
    -0.000339031, -0.000371456, -0.000404358, -0.000438213,
    -0.000472546, -0.000507355, -0.000542164, -0.000576973,
    -0.000611782, -0.000646591, -0.000680923, -0.000714302,
    -0.000747204, -0.000779152, -0.000809669, -0.000838757,
    -0.000866413, -0.000891685, -0.000915051, -0.000935555,
    -0.000954151, -0.000968933, -0.000980854, -0.000989437,
    -0.000994205, -0.000995159, -0.000991821, -0.000983715,
    0.000971317,  0.000953674,  0.000930786,  0.000902653,
    0.000868797,  0.000829220,  0.000783920,  0.000731945,
    0.000674248,  0.000610352,  0.000539303,  0.000462532,
    0.000378609,  0.000288486,  0.000191689,  0.000088215,
    -0.000021458, -0.000137329, -0.000259876, -0.000388145,
    -0.000522137, -0.000661850, -0.000806808, -0.000956535,
    -0.001111031, -0.001269817, -0.001432419, -0.001597881,
    -0.001766682, -0.001937389, -0.002110004, -0.002283096,
    -0.002457142, -0.002630711, -0.002803326, -0.002974033,
    -0.003141880, -0.003306866, -0.003467083, -0.003622532,
    -0.003771782, -0.003914356, -0.004048824, -0.004174709,
    -0.004290581, -0.004395962, -0.004489899, -0.004570484,
    -0.004638195, -0.004691124, -0.004728317, -0.004748821,
    -0.004752159, -0.004737377, -0.004703045, -0.004649162,
    -0.004573822, -0.004477024, -0.004357815, -0.004215240,
    -0.004049301, -0.003858566, -0.003643036, -0.003401756,
    0.003134727,  0.002841473,  0.002521515,  0.002174854,
    0.001800537,  0.001399517,  0.000971317,  0.000515938,
    0.000033379, -0.000475883, -0.001011848, -0.001573563,
    -0.002161503, -0.002774239, -0.003411293, -0.004072189,
    -0.004756451, -0.005462170, -0.006189346, -0.006937027,
    -0.007703304, -0.008487225, -0.009287834, -0.010103703,
    -0.010933399, -0.011775017, -0.012627602, -0.013489246,
    -0.014358521, -0.015233517, -0.016112804, -0.016994476,
    -0.017876148, -0.018756866, -0.019634247, -0.020506859,
    -0.021372318, -0.022228718, -0.023074150, -0.023907185,
    -0.024725437, -0.025527000, -0.026310921, -0.027073860,
    -0.027815342, -0.028532982, -0.029224873, -0.029890060,
    -0.030526638, -0.031132698, -0.031706810, -0.032248020,
    -0.032754898, -0.033225536, -0.033659935, -0.034055710,
    -0.034412861, -0.034730434, -0.035007000, -0.035242081,
    -0.035435200, -0.035586357, -0.035694122, -0.035758972,
    0.035780907,  0.035758972,  0.035694122,  0.035586357,
    0.035435200,  0.035242081,  0.035007000,  0.034730434,
    0.034412861,  0.034055710,  0.033659935,  0.033225536,
    0.032754898,  0.032248020,  0.031706810,  0.031132698,
    0.030526638,  0.029890060,  0.029224873,  0.028532982,
    0.027815342,  0.027073860,  0.026310921,  0.025527000,
    0.024725437,  0.023907185,  0.023074150,  0.022228718,
    0.021372318,  0.020506859,  0.019634247,  0.018756866,
    0.017876148,  0.016994476,  0.016112804,  0.015233517,
    0.014358521,  0.013489246,  0.012627602,  0.011775017,
    0.010933399,  0.010103703,  0.009287834,  0.008487225,
    0.007703304,  0.006937027,  0.006189346,  0.005462170,
    0.004756451,  0.004072189,  0.003411293,  0.002774239,
    0.002161503,  0.001573563,  0.001011848,  0.000475883,
    -0.000033379, -0.000515938, -0.000971317, -0.001399517,
    -0.001800537, -0.002174854, -0.002521515, -0.002841473,
    0.003134727,  0.003401756,  0.003643036,  0.003858566,
    0.004049301,  0.004215240,  0.004357815,  0.004477024,
    0.004573822,  0.004649162,  0.004703045,  0.004737377,
    0.004752159,  0.004748821,  0.004728317,  0.004691124,
    0.004638195,  0.004570484,  0.004489899,  0.004395962,
    0.004290581,  0.004174709,  0.004048824,  0.003914356,
    0.003771782,  0.003622532,  0.003467083,  0.003306866,
    0.003141880,  0.002974033,  0.002803326,  0.002630711,
    0.002457142,  0.002283096,  0.002110004,  0.001937389,
    0.001766682,  0.001597881,  0.001432419,  0.001269817,
    0.001111031,  0.000956535,  0.000806808,  0.000661850,
    0.000522137,  0.000388145,  0.000259876,  0.000137329,
    0.000021458, -0.000088215, -0.000191689, -0.000288486,
    -0.000378609, -0.000462532, -0.000539303, -0.000610352,
    -0.000674248, -0.000731945, -0.000783920, -0.000829220,
    -0.000868797, -0.000902653, -0.000930786, -0.000953674,
    0.000971317,  0.000983715,  0.000991821,  0.000995159,
    0.000994205,  0.000989437,  0.000980854,  0.000968933,
    0.000954151,  0.000935555,  0.000915051,  0.000891685,
    0.000866413,  0.000838757,  0.000809669,  0.000779152,
    0.000747204,  0.000714302,  0.000680923,  0.000646591,
    0.000611782,  0.000576973,  0.000542164,  0.000507355,
    0.000472546,  0.000438213,  0.000404358,  0.000371456,
    0.000339031,  0.000307560,  0.000277042,  0.000247478,
    0.000218868,  0.000191212,  0.000165462,  0.000140190,
    0.000116348,  0.000093937,  0.000072956,  0.000052929,
    0.000034332,  0.000017166,  0.000000954, -0.000013828,
    -0.000027180, -0.000039577, -0.000050545, -0.000060558,
    -0.000069618, -0.000077724, -0.000084400, -0.000090122,
    -0.000095367, -0.000099182, -0.000102520, -0.000105381,
    -0.000106812, -0.000108242, -0.000108719, -0.000108719,
    -0.000108242, -0.000107288, -0.000105858, -0.000103951,
    0.000101566,  0.000099182,  0.000096321,  0.000093460,
    0.000090599,  0.000087261,  0.000083923,  0.000080585,
    0.000076771,  0.000073433,  0.000070095,  0.000066280,
    0.000062943,  0.000059605,  0.000055790,  0.000052929,
    0.000049591,  0.000046253,  0.000043392,  0.000040531,
    0.000037670,  0.000034809,  0.000032425,  0.000030041,
    0.000027657,  0.000025272,  0.000023365,  0.000021458,
    0.000019550,  0.000018120,  0.000016689,  0.000014782,
    0.000013828,  0.000012398,  0.000011444,  0.000010014,
    0.000009060,  0.000008106,  0.000007629,  0.000006676,
    0.000006199,  0.000005245,  0.000004768,  0.000004292,
    0.000003815,  0.000003338,  0.000003338,  0.000002861,
    0.000002384,  0.000002384,  0.000001907,  0.000001907,
    0.000001431,  0.000001431,  0.000000954,  0.000000954,
    0.000000954,  0.000000954,  0.000000477,  0.000000477,
    0.000000477,  0.000000477,  0.000000477,  0.000000477,
]

DECODER_C_COEF = [
    0.000000000, -0.000015259, -0.000015259, -0.000015259,
    -0.000015259, -0.000015259, -0.000015259, -0.000030518,
    -0.000030518, -0.000030518, -0.000030518, -0.000045776,
    -0.000045776, -0.000061035, -0.000061035, -0.000076294,
    -0.000076294, -0.000091553, -0.000106812, -0.000106812,
    -0.000122070, -0.000137329, -0.000152588, -0.000167847,
    -0.000198364, -0.000213623, -0.000244141, -0.000259399,
    -0.000289917, -0.000320435, -0.000366211, -0.000396729,
    -0.000442505, -0.000473022, -0.000534058, -0.000579834,
    -0.000625610, -0.000686646, -0.000747681, -0.000808716,
    -0.000885010, -0.000961304, -0.001037598, -0.001113892,
    -0.001205444, -0.001296997, -0.001388550, -0.001480103,
    -0.001586914, -0.001693726, -0.001785278, -0.001907349,
    -0.002014160, -0.002120972, -0.002243042, -0.002349854,
    -0.002456665, -0.002578735, -0.002685547, -0.002792358,
    -0.002899170, -0.002990723, -0.003082275, -0.003173828,
    0.003250122,  0.003326416,  0.003387451,  0.003433228,
    0.003463745,  0.003479004,  0.003479004,  0.003463745,
    0.003417969,  0.003372192,  0.003280640,  0.003173828,
    0.003051758,  0.002883911,  0.002700806,  0.002487183,
    0.002227783,  0.001937866,  0.001617432,  0.001266479,
    0.000869751,  0.000442505, -0.000030518, -0.000549316,
    -0.001098633, -0.001693726, -0.002334595, -0.003005981,
    -0.003723145, -0.004486084, -0.005294800, -0.006118774,
    -0.007003784, -0.007919312, -0.008865356, -0.009841919,
    -0.010848999, -0.011886597, -0.012939453, -0.014022827,
    -0.015121460, -0.016235352, -0.017349243, -0.018463135,
    -0.019577026, -0.020690918, -0.021789551, -0.022857666,
    -0.023910522, -0.024932861, -0.025909424, -0.026840210,
    -0.027725220, -0.028533936, -0.029281616, -0.029937744,
    -0.030532837, -0.031005859, -0.031387329, -0.031661987,
    -0.031814575, -0.031845093, -0.031738281, -0.031478882,
    0.031082153,  0.030517578,  0.029785156,  0.028884888,
    0.027801514,  0.026535034,  0.025085449,  0.023422241,
    0.021575928,  0.019531250,  0.017257690,  0.014801025,
    0.012115479,  0.009231567,  0.006134033,  0.002822876,
    -0.000686646, -0.004394531, -0.008316040, -0.012420654,
    -0.016708374, -0.021179199, -0.025817871, -0.030609131,
    -0.035552979, -0.040634155, -0.045837402, -0.051132202,
    -0.056533813, -0.061996460, -0.067520142, -0.073059082,
    -0.078628540, -0.084182739, -0.089706421, -0.095169067,
    -0.100540161, -0.105819702, -0.110946655, -0.115921021,
    -0.120697021, -0.125259399, -0.129562378, -0.133590698,
    -0.137298584, -0.140670776, -0.143676758, -0.146255493,
    -0.148422241, -0.150115967, -0.151306152, -0.151962280,
    -0.152069092, -0.151596069, -0.150497437, -0.148773193,
    -0.146362305, -0.143264771, -0.139450073, -0.134887695,
    -0.129577637, -0.123474121, -0.116577148, -0.108856201,
    0.100311279,  0.090927124,  0.080688477,  0.069595337,
    0.057617187,  0.044784546,  0.031082153,  0.016510010,
    0.001068115, -0.015228271, -0.032379150, -0.050354004,
    -0.069168091, -0.088775635, -0.109161377, -0.130310059,
    -0.152206421, -0.174789429, -0.198059082, -0.221984863,
    -0.246505737, -0.271591187, -0.297210693, -0.323318481,
    -0.349868774, -0.376800537, -0.404083252, -0.431655884,
    -0.459472656, -0.487472534, -0.515609741, -0.543823242,
    -0.572036743, -0.600219727, -0.628295898, -0.656219482,
    -0.683914185, -0.711318970, -0.738372803, -0.765029907,
    -0.791213989, -0.816864014, -0.841949463, -0.866363525,
    -0.890090942, -0.913055420, -0.935195923, -0.956481934,
    -0.976852417, -0.996246338, -1.014617920, -1.031936646,
    -1.048156738, -1.063217163, -1.077117920, -1.089782715,
    -1.101211548, -1.111373901, -1.120223999, -1.127746582,
    -1.133926392, -1.138763428, -1.142211914, -1.144287109,
    1.144989014,  1.144287109,  1.142211914,  1.138763428,
    1.133926392,  1.127746582,  1.120223999,  1.111373901,
    1.101211548,  1.089782715,  1.077117920,  1.063217163,
    1.048156738,  1.031936646,  1.014617920,  0.996246338,
    0.976852417,  0.956481934,  0.935195923,  0.913055420,
    0.890090942,  0.866363525,  0.841949463,  0.816864014,
    0.791213989,  0.765029907,  0.738372803,  0.711318970,
    0.683914185,  0.656219482,  0.628295898,  0.600219727,
    0.572036743,  0.543823242,  0.515609741,  0.487472534,
    0.459472656,  0.431655884,  0.404083252,  0.376800537,
    0.349868774,  0.323318481,  0.297210693,  0.271591187,
    0.246505737,  0.221984863,  0.198059082,  0.174789429,
    0.152206421,  0.130310059,  0.109161377,  0.088775635,
    0.069168091,  0.050354004,  0.032379150,  0.015228271,
    -0.001068115, -0.016510010, -0.031082153, -0.044784546,
    -0.057617187, -0.069595337, -0.080688477, -0.090927124,
    0.100311279,  0.108856201,  0.116577148,  0.123474121,
    0.129577637,  0.134887695,  0.139450073,  0.143264771,
    0.146362305,  0.148773193,  0.150497437,  0.151596069,
    0.152069092,  0.151962280,  0.151306152,  0.150115967,
    0.148422241,  0.146255493,  0.143676758,  0.140670776,
    0.137298584,  0.133590698,  0.129562378,  0.125259399,
    0.120697021,  0.115921021,  0.110946655,  0.105819702,
    0.100540161,  0.095169067,  0.089706421,  0.084182739,
    0.078628540,  0.073059082,  0.067520142,  0.061996460,
    0.056533813,  0.051132202,  0.045837402,  0.040634155,
    0.035552979,  0.030609131,  0.025817871,  0.021179199,
    0.016708374,  0.012420654,  0.008316040,  0.004394531,
    0.000686646, -0.002822876, -0.006134033, -0.009231567,
    -0.012115479, -0.014801025, -0.017257690, -0.019531250,
    -0.021575928, -0.023422241, -0.025085449, -0.026535034,
    -0.027801514, -0.028884888, -0.029785156, -0.030517578,
    0.031082153,  0.031478882,  0.031738281,  0.031845093,
    0.031814575,  0.031661987,  0.031387329,  0.031005859,
    0.030532837,  0.029937744,  0.029281616,  0.028533936,
    0.027725220,  0.026840210,  0.025909424,  0.024932861,
    0.023910522,  0.022857666,  0.021789551,  0.020690918,
    0.019577026,  0.018463135,  0.017349243,  0.016235352,
    0.015121460,  0.014022827,  0.012939453,  0.011886597,
    0.010848999,  0.009841919,  0.008865356,  0.007919312,
    0.007003784,  0.006118774,  0.005294800,  0.004486084,
    0.003723145,  0.003005981,  0.002334595,  0.001693726,
    0.001098633,  0.000549316,  0.000030518, -0.000442505,
    -0.000869751, -0.001266479, -0.001617432, -0.001937866,
    -0.002227783, -0.002487183, -0.002700806, -0.002883911,
    -0.003051758, -0.003173828, -0.003280640, -0.003372192,
    -0.003417969, -0.003463745, -0.003479004, -0.003479004,
    -0.003463745, -0.003433228, -0.003387451, -0.003326416,
    0.003250122,  0.003173828,  0.003082275,  0.002990723,
    0.002899170,  0.002792358,  0.002685547,  0.002578735,
    0.002456665,  0.002349854,  0.002243042,  0.002120972,
    0.002014160,  0.001907349,  0.001785278,  0.001693726,
    0.001586914,  0.001480103,  0.001388550,  0.001296997,
    0.001205444,  0.001113892,  0.001037598,  0.000961304,
    0.000885010,  0.000808716,  0.000747681,  0.000686646,
    0.000625610,  0.000579834,  0.000534058,  0.000473022,
    0.000442505,  0.000396729,  0.000366211,  0.000320435,
    0.000289917,  0.000259399,  0.000244141,  0.000213623,
    0.000198364,  0.000167847,  0.000152588,  0.000137329,
    0.000122070,  0.000106812,  0.000106812,  0.000091553,
    0.000076294,  0.000076294,  0.000061035,  0.000061035,
    0.000045776,  0.000045776,  0.000030518,  0.000030518,
    0.000030518,  0.000030518,  0.000015259,  0.000015259,
    0.000015259,  0.000015259,  0.000015259,  0.000015259,
]

assert len(ENCODER_C_COEF) == 512
assert len(DECODER_C_COEF) == 512

# フィルタ係数に変換
ENCODER_FILTER_COEF = [ENCODER_C_COEF[i] if ((i // 64) % 2 == 0) else -ENCODER_C_COEF[i] for i in range(512)]
DECODER_FILTER_COEF = [DECODER_C_COEF[i] if ((i // 64) % 2 == 0) else -DECODER_C_COEF[i] for i in range(512)]

def mp3_analysis_filter(data):
    out = []
    for k in range(32):
        coef = ENCODER_FILTER_COEF * np.cos(np.pi / 32 * (k + 1/2) * (np.arange(0, 512) - 16))
        # 畳み込み
        sig = np.convolve(data, coef, 'full')
        out.append(sig)
    return out

def mp3_synthesis_filter(data):
    out = np.zeros(len(data[0]))
    for k, sig in enumerate(data):
        # 畳み込み
        coef = DECODER_FILTER_COEF * np.cos(np.pi / 32 * (k + 1/2) * (np.arange(0, 512) + 16))
        out += np.convolve(sig, coef, 'same')
    return out

def mp3_decimation(bands):
    out = []
    # 各バンドで間引く
    for band in bands:
        out.append(band[::32])
    return out

def mp3_interpolation(bands):
    out = []
    outlen = 32 * len(bands[0])
    # 各バンドで補間
    for band in bands:
        interp = np.zeros(outlen)
        interp[::32] = band
        out.append(interp)
    return out

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

def plot_frequency_inversion():
    '''
    フィルタバンク（バンドパスフィルタ）の間でピークを持つ正弦波を入力したときの
    分析フィルタバンクの周波数特性をプロット
    奇数バンドで周波数特性を逆転させている（(-1)**nを掛けている）理由を探る
    '''
    NUM_SAMPLES = 1024 * 4

    for k in range(31):
        data = np.sin(2.0 * np.pi * smpls * ((k + 1) / 64.0))
        analy = mp3_analysis_filter(data)
        decim = mp3_decimation(analy)

        plt.cla()
        for i in np.arange(k, k + 2):
            spec = np.fft.fft(decim[i], norm='forward')[:len(decim[i])//2 + 1]
            if i % 2 == 1:
                # 奇数バンド：(-1)**n を掛けて周波数特性を逆転させる
                freqinv = decim[i] * ((-1) ** np.arange(0, len(decim[i])))
                invspec = np.fft.fft(freqinv, norm='forward')[:len(decim[i])//2 + 1]
                plt.plot(20 * np.log10(np.abs(invspec)), label=f'bank {i} (freq inverted)')
                plt.plot(20 * np.log10(np.abs(spec)), label=f'bank {i}', linestyle='--')
            else:
                plt.plot(20 * np.log10(np.abs(spec)), label=f'bank {i}')
        plt.title(f'MP3 analysis filterbank output for {k + 1}/64 Hz sin wave')
        plt.ylim((-70, 0))
        plt.xlabel('bin')
        plt.ylabel('amplitude (dB)')
        plt.grid()
        plt.legend()
        plt.savefig(f'analysis_filter_output_{k + 1}_64Hz_sin.pdf')

def plot_analysis_synthesis_filter_impulse_resonse():
    '''
    フィルタバンクで分析合成するときのインパルス応答を計算・プロット
    '''
    analy_coefs = []
    synth_coefs = []
    for k in range(32):
        analy_coef = ENCODER_FILTER_COEF * np.cos(np.pi / 32 * (k + 1/2) * (np.arange(0, 512) - 16))
        synth_coef = DECODER_FILTER_COEF * np.cos(np.pi / 32 * (k + 1/2) * (np.arange(0, 512) + 16))
        analy_coefs.append(analy_coef)
        synth_coefs.append(synth_coef)
    sums = np.zeros(1023)
    for m in range(1023):
        for i in np.arange(max(0, m - 511), min(511, m) + 1):
            for k in range(32):
                sums[m] += analy_coefs[k][i] * synth_coefs[k][m - i]
    plt.cla()
    plt.plot(20.0 * np.log10(sums))
    plt.title('impulse response of MP3 analysys-synthesis filter')
    plt.xlabel('sample')
    plt.ylabel('amplitude (dB)')
    plt.xticks(np.arange(0, 1025, 128))
    plt.ylim((-250, 50))
    plt.grid()
    plt.savefig(f'impluse_responce_of_MP3_analysis_synthesis_filter.pdf')

if __name__ == '__main__':
    NUM_SAMPLES = 1024 * 4
    smpls = np.arange(0, NUM_SAMPLES)
    data = np.sin(2.0 * np.pi * 0.01 * smpls / 20.0) + np.cos(2.0 * np.pi * 0.3 * smpls / 20.0)

    analy = mp3_analysis_filter(data)
    decim = mp3_decimation(analy)
    intrp = mp3_interpolation(decim)
    synth = mp3_synthesis_filter(intrp)[512 // 2 + 1:][:NUM_SAMPLES] # 畳み込み係数の半分だけ遅延する（線形位相特性）

    rmse = np.mean((data - synth) ** 2.0) ** 0.5
    print(20.0 * np.log10(rmse))

    plot_frequency_inversion()
    plot_analysis_synthesis_filter_impulse_resonse()
