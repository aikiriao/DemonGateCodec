#ifndef DGPLAYER_H_INCLUDED
#define DGPLAYER_H_INCLUDED

#include <stdint.h>

/* 出力要求コールバック */
typedef void (*DGSampleRequestCallback)(
        int32_t **buffer, uint32_t num_channels, uint32_t num_samples);

/* プレイヤー初期化コンフィグ */
struct DGPlayerConfig {
    uint32_t sampling_rate;
    uint16_t num_channels;
    uint16_t bits_per_sample;
    DGSampleRequestCallback sample_request_callback;
};

#ifdef __cplusplus
extern "C" {
#endif

/* 初期化 この関数内でデバイスドライバの初期化を行い、再生開始 */
void DGPlayer_Initialize(const struct DGPlayerConfig *config);

/* 終了 初期化したときのリソースの開放はここで */
void DGPlayer_Finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* DGPLAYER_H_INCLUDED */
