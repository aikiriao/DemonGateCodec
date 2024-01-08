#include "dg_player.h"
#include <dg_decoder.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* 出力要求コールバック */
static void DGPlayer_SampleRequestCallback(int32_t **buffer, uint32_t num_channels, uint32_t num_samples);
/* 終了処理 */
static void exit_dg_player(void);

/* 再生制御のためのグローバル変数 */
static struct DGFormatInformation format;
static uint32_t output_samples = 0;
static int16_t *decode_buffer[MP3_MAX_NUM_CHANNELS] = { NULL, };
static uint32_t num_buffered_samples = 0;
static uint32_t buffer_pos = 0;
static uint32_t data_size = 0;
static uint8_t *data = NULL;
static uint32_t decode_offset = 0;
static struct DGDecoder *decoder = NULL;

/* メインエントリ */
int main(int argc, char **argv)
{
    uint32_t i;
    enum DGDecoderDecodeResult ret;
    struct DGPlayerConfig player_config;

    /* 引数チェック 間違えたら使用方法を提示 */
    if (argc != 2) {
        printf("Usage: %s MP3FILE \n", argv[0]);
        return 1;
    }

    /* ファイルのロード */
    {
        struct stat fstat;
        FILE *fp;
        const char *filename = argv[1];

        /* ファイルオープン */
        if ((fp = fopen(filename, "rb")) == NULL) {
            fprintf(stderr, "Failed to open %s \n", filename);
            return 1;
        }

        /* 入力ファイルのサイズ取得 / バッファ領域割り当て */
        stat(filename, &fstat);
        data_size = (uint32_t)fstat.st_size;
        data = (uint8_t *)malloc(data_size);

        /* バッファ領域にデータをロード */
        if (fread(data, sizeof(uint8_t), data_size, fp) < data_size) {
            fprintf(stderr, "Failed to load %s data \n", filename);
            return 1;
        }

        fclose(fp);
    }

    /* チャンネル数の特定 */
    if ((ret = DGDecoder_GetFormatInformation(data, data_size, &format)) != DGDECODER_DECODERESULT_OK) {
        fprintf(stderr, "Failed to detect channel size.\n");
        return 1;
    }

    /* デコーダ作成 */
    decoder = DGDecoder_Create(NULL, 0);

    /* バッファ領域確保 */
    for (i = 0; i < format.num_channels; i++) {
        decode_buffer[i] = malloc(sizeof(int16_t) * MP3_NUM_SAMPLES_PER_FRAME);
    }

    /* プレイヤー初期化 */
    player_config.sampling_rate = format.sampling_rate;
    player_config.num_channels = format.num_channels;
    player_config.bits_per_sample = 16;
    player_config.sample_request_callback = DGPlayer_SampleRequestCallback;
    DGPlayer_Initialize(&player_config);

    /* この後はコールバック要求により進む */
    while (1) { ; }

    return 0;
}

/* 出力要求コールバック */
static void DGPlayer_SampleRequestCallback(int32_t **buffer, uint32_t num_channels, uint32_t num_samples)
{
    uint32_t ch, smpl;

    for (smpl = 0; smpl < num_samples; smpl++) {
        /* バッファを使い切ったら即時にデコード */
        if (buffer_pos >= num_buffered_samples) {
            size_t decode_size;
            struct MP3FrameHeader header;
            struct MP3SideInformation side_info;
            if (DGDecoder_DecodeFrame(decoder,
                        &data[decode_offset], data_size - decode_offset,
                        &header, &side_info, decode_buffer, num_channels, MP3_NUM_SAMPLES_PER_FRAME,
                        &decode_size) != DGDECODER_DECODERESULT_OK) {
                fprintf(stderr, "decoding error! \n");
                exit(1);
            }
            buffer_pos = 0;
            decode_offset += decode_size;
            num_buffered_samples = MP3_NUM_SAMPLES_PER_FRAME;
        }

        /* 出力用バッファ領域にコピー */
        for (ch = 0; ch < num_channels; ch++) {
            buffer[ch][smpl] = decode_buffer[ch][buffer_pos];
        }
        buffer_pos++;
        output_samples++;

        /* 再生終了次第終了処理へ */
        if (output_samples >= format.num_samples) {
            exit_dg_player();
        }
    }

    /* 進捗表示 */
    printf("playing... %7.3f / %7.3f \r",
            (double)output_samples / format.sampling_rate, (double)format.num_samples / format.sampling_rate);
    fflush(stdout);
}

/* 終了処理 */
static void exit_dg_player(void)
{
    uint32_t i;

    DGPlayer_Finalize();

    for (i = 0; i < format.num_channels; i++) {
        free(decode_buffer[i]);
    }
    DGDecoder_Destroy(decoder);
    free(data);

    exit(0);
}
