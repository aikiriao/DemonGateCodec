#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <gtest/gtest.h>

/* テスト対象のモジュール */
extern "C" {
#include "../../libs/dg_decoder/src/dg_decoder.c"
}

#include <wav.h>

/* デコードテスト */
TEST(DecoderTest, DecodeTest)
{
    int32_t i;
    struct DecodeTestCase {
        const char *reference_wav_filename;
        const char *input_mpeg_filename;
    };

    /* dist10のリファレンス出力はReleaseビルドしたものを使用（Debugでは未初期領域アクセスにより高域が消える） */
    static const struct DecodeTestCase decode_testcase_list[] = {
        {             "y004_320_encdist10_decdist10.wav",             "y004_320_encdist10.mpg" },
        {             "y004_128_encdist10_decdist10.wav",             "y004_128_encdist10.mpg" },
        {              "y004_32_encdist10_decdist10.wav",              "y004_32_encdist10.mpg" },
        {             "y004_320_encffmpeg_decdist10.wav",             "y004_320_encffmpeg.mp3" },
        {             "y004_128_encffmpeg_decdist10.wav",             "y004_128_encffmpeg.mp3" },
        {              "y004_32_encffmpeg_decdist10.wav",              "y004_32_encffmpeg.mp3" },
        {               "y004_320_encgogo_decdist10.wav",               "y004_320_encgogo.mp3" },
        {               "y004_128_encgogo_decdist10.wav",               "y004_128_encgogo.mp3" },
        {                "y004_64_encgogo_decdist10.wav",                "y004_64_encgogo.mp3" },
        {  "alphabet02all_01_32_encffmpeg_decdist10.wav",  "alphabet02all_01_32_encffmpeg.mp3" },
        { "alphabet02all_01_128_encffmpeg_decdist10.wav", "alphabet02all_01_128_encffmpeg.mp3" },
        { "alphabet02all_01_320_encffmpeg_decdist10.wav", "alphabet02all_01_320_encffmpeg.mp3" },
    };
    const int32_t num_decode_testcase = sizeof(decode_testcase_list) / sizeof(decode_testcase_list[0]);

    for (i = 0; i < num_decode_testcase; i++) {
        FILE *in_fp;
        uint8_t *data;
        size_t data_size;
        struct stat fstat;
        struct WAVFile *inwav;
        struct DGFormatInformation format;

        const struct DecodeTestCase *pcase = &decode_testcase_list[i];

        inwav = WAV_CreateFromFile(pcase->reference_wav_filename);
        ASSERT_TRUE(inwav != NULL);

        /* 入力ファイルオープン */
        in_fp = fopen(pcase->input_mpeg_filename, "rb");
        /* 入力ファイルのサイズ取得 / バッファ領域割り当て */
        stat(pcase->input_mpeg_filename, &fstat);
        data_size = fstat.st_size;
        data = (uint8_t *)malloc(data_size);
        /* バッファ領域にデータをロード */
        fread(data, sizeof(uint8_t), data_size, in_fp);
        fclose(in_fp);

        {
            struct DGDecoder *decoder;
            int16_t *decoded_buffer[MP3_MAX_NUM_CHANNELS];
            int32_t ch, smpl;
            int32_t num_buffer_samples, num_buffer_channels, decoded_samples;
            int32_t max_error;
            enum DGDecoderDecodeResult ret;

            ret = DGDecoder_GetFormatInformation(data, data_size, &format);
            ASSERT_EQ(DGDECODER_DECODERESULT_OK, ret);
            num_buffer_samples = format.num_samples;
            num_buffer_channels = format.num_channels;

            ASSERT_EQ(inwav->format.num_channels, num_buffer_channels);
            /* 本当は一致が良いが妥協 */
            ASSERT_TRUE(num_buffer_samples >= inwav->format.num_samples);

            for (ch = 0; ch < num_buffer_channels; ch++) {
                decoded_buffer[ch] = (int16_t *)malloc(sizeof(int16_t) * num_buffer_samples);
            }

            decoder = DGDecoder_Create(NULL, 0);
            ASSERT_TRUE(decoder != NULL);

            ret = DGDecoder_DecodeWhole(decoder, data, data_size,
                    decoded_buffer, num_buffer_channels, num_buffer_samples, &decoded_samples);
            ASSERT_EQ(DGDECODER_DECODERESULT_OK, ret);

            DGDecoder_Destroy(decoder);

            ASSERT_EQ(num_buffer_samples, decoded_samples);
            /* 本当は一致が良いが妥協 */
            ASSERT_TRUE(decoded_samples >= inwav->format.num_samples);

            max_error = 0;
            int max_index = -1;
            for (ch = 0; ch < inwav->format.num_channels; ch++) {
                /* 遅延サンプル分(=1057)除いて比較（モノラルデータでは、dist10のデコード結果末尾で成分が発生...） */
                for (smpl = 0; smpl < inwav->format.num_samples - 1057; smpl++) {
                    const int32_t ref = WAVFile_PCM(inwav, smpl, ch) >> 16;
                    const int32_t error = abs(decoded_buffer[ch][smpl] - ref);
                    if (error > max_error) {
                        max_error = error;
                        max_index = smpl;
                    }
                }
            }

            /* 浮動小数の演算順序の差でビットパーフェクトは不可能なので、マージンを設ける */
            EXPECT_TRUE(max_error <= 1);

            for (ch = 0; ch < num_buffer_channels; ch++) {
                free(decoded_buffer[ch]);
            }
        }

        free(data);

        WAV_Destroy(inwav);
    }
}
