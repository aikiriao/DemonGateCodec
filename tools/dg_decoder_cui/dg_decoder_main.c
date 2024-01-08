#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

#include <dg_decoder.h>
#include <wav.h>

/* メインエントリ */
int main(int argc, char **argv)
{
    FILE *in_fp;
    uint8_t *data;
    size_t data_size;
    struct stat fstat;
    const char *in_filename, *out_filename;

    /* 引数が足りないときはヘルプを表示 */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s INFILE.mp3 OUTFILE.wav", argv[0]);
        exit(1);
    }

    /* ファイル名取得 */
    in_filename = argv[1];
    out_filename = argv[2];

    /* 入力ファイルオープン */
    in_fp = fopen(in_filename, "rb");
    /* 入力ファイルのサイズ取得 / バッファ領域割り当て */
    stat(in_filename, &fstat);
    data_size = (size_t)fstat.st_size;
    data = (uint8_t *)malloc(data_size);
    /* バッファ領域にデータをロード */
    fread(data, sizeof(uint8_t), data_size, in_fp);
    fclose(in_fp);

    {
        struct DGDecoder *decoder;
        int16_t *decoded_buffer[MP3_MAX_NUM_CHANNELS];
        int32_t ch, decoded_samples;
        enum DGDecoderDecodeResult ret;
        struct DGFormatInformation format;

        /* フォーマット取得 */
        if ((ret = DGDecoder_GetFormatInformation(data, data_size, &format)) != DGDECODER_DECODERESULT_OK) {
            fprintf(stderr, "failed to decode.");
            return -1;
        }

        /* バッファ領域確保 */
        for (ch = 0; ch < format.num_channels; ch++) {
            decoded_buffer[ch] = (int16_t *)malloc(sizeof(int16_t) * (size_t)format.num_samples);
        }

        /* デコーダ作成・デコード */
        decoder = DGDecoder_Create(NULL, 0);
        if ((ret = DGDecoder_DecodeWhole(decoder, data, data_size,
            decoded_buffer, format.num_channels, format.num_samples, &decoded_samples)) != DGDECODER_DECODERESULT_OK) {
            fprintf(stderr, "failed to decode.");
            return -1;
        }
        DGDecoder_Destroy(decoder);

        /* wavファイル書き出し */
        {
            int32_t smpl;
            struct WAVFileFormat wav_format;
            struct WAVFile *outwav;

            /* wavのフォーマット作成 */
            wav_format.bits_per_sample = 16;
            wav_format.data_format = WAV_DATA_FORMAT_PCM;
            wav_format.num_channels = format.num_channels;
            wav_format.num_samples = format.num_samples;
            wav_format.sampling_rate = format.sampling_rate;

            outwav = WAV_Create(&wav_format);
            for (smpl = 0; smpl < decoded_samples; smpl++) {
                for (ch = 0; ch < format.num_channels; ch++) {
                    WAVFile_PCM(outwav, smpl, ch) = (int32_t)decoded_buffer[ch][smpl] << 16;
                }
            }
            WAV_WriteToFile(out_filename, outwav);
            WAV_Destroy(outwav);
        }

        for (ch = 0; ch < format.num_channels; ch++) {
            free(decoded_buffer[ch]);
        }
    }

    free(data);

    return 0;
}
