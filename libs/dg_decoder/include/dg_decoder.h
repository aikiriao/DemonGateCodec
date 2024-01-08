#ifndef DGDECODER_H_INCLUDED
#define DGDECODER_H_INCLUDED

#include <stdint.h>
#include <stddef.h>

#define MPEG_VERSION_1 1 /*!< MPEG1 */
#define MPEG_VERSION_2 0 /*!< MPEG2(LSF, Low Sampling Frequency) */
#define MP3_MAX_NUM_CHANNELS 2 /*!< 最大チャンネル数（ステレオ） */
#define MP3_NUM_SAMPLES_PER_FRAME 1152 /*!< 1フレームあたりサンプル数 */

/*! デコード結果型 */
enum DGDecoderDecodeResult {
    DGDECODER_DECODERESULT_OK = 0, /*!< OK */
    DGDECODER_DECODERESULT_EOS, /*!< ストリーム終端に達した */
    DGDECODER_DECODERESULT_INVALID_ARGUMENT, /*!< 不正な引数 */
    DGDECODER_DECODERESULT_NG /*!< 分類不能なエラー */
};

/*! ブロックタイプ */
enum MP3BlockType {
    MP3_BLOCKTYPE_NORMAL = 0, /*!< 通常の窓(long) */
    MP3_BLOCKTYPE_START = 1, /*!< ショートブロックの開始 */
    MP3_BLOCKTYPE_SHORT = 2, /*!< ショートブロック */
    MP3_BLOCKTYPE_STOP = 3 /*!< ショートブロックの終端 */
};

/*! チャンネルモード */
enum MP3ChannelMode {
    MP3_CHANNELMODE_STEREO = 0, /*!< ステレオ */
    MP3_CHANNELMODE_JOINTSTEREO = 1, /*!< ジョイントステレオ */
    MP3_CHANNELMODE_DUALCHANNEL = 2, /*!< デュアルチャンネル */
    MP3_CHANNELMODE_MONORAL = 3 /*!< モノラル */
};

/*! フレームヘッダ情報 */
struct MP3FrameHeader {
    uint8_t version; /*!< バージョン(0:MPEG_PHASE2_LSF, 1:MPEG1 Audio) 1bit */
    uint8_t layer; /*!< レイヤー(1:Layer1, 2:Layer2, 3:Layer3) 2bit */
    uint8_t error_protection; /*!< エラー保護するかどうか 1bit */
    uint8_t bitrate_index; /*!< ビットレートテーブルインデックス 4bit */
    uint8_t sampling_frequency; /*!< サンプリングレート(0:44100, 1:48000, 2:32000) 2bit */
    uint8_t padding; /*!< フレームサイズ調整のパディングバイトがあるかどうか 1bit */
    uint8_t extension; /*!< ユーザーが使用するデータ 1bit */
    uint8_t mode; /*!< チャンネルモード(0:ステレオ, 1:ジョイントステレオ, 2:デュアルチャンネル, 3:モノラル) 2bit */
    uint8_t mode_ext; /*!< 1bit目: インテンシティステレオか否か, 2bit目: MSステレオか否か 2bit */
    uint8_t copyright; /*!< 1であれば不正コピーを意味（dist10では未使用） 1bit */
    uint8_t original; /*!< 1であれば原盤（dist10では未使用） 1bit */
    uint8_t emphasis; /*!< エンファシス(0:None, 1:50/15ms, 2:Reserved, 3:CCITT J.17)（dist10では未使用） 2bit */
};

/*! グラニュール情報 */
struct MP3GranuleInformation {
    uint16_t part2_3_length; /*!< 12bit スケールファクタとハフマン符号化されたビット数の和 */
    uint16_t big_values; /*!< 9bit この値の2倍がbigvalue_bandのサンプル数 */
    uint8_t global_gain; /*!< 8bit 量子化ステップを表すパラメータ */
    uint8_t scalefac_compress; /*!< 4bit スケールファクタのビット幅のテーブルインデックス */
    uint8_t window_switching_flag; /*!< 1bit normalなら0, normalでないなら1 */
    uint8_t block_type; /*!< 2bit(window_switching_flag == 1) 窓関数タイプ(0:normal, 1:start, 2:short, 3:stop) l3psy.h にマクロ定義あり */
    uint8_t mixed_block_flag; /*!< 1bit(window_switching_flag == 1) mix typeのとき1（dist10では常に0） */
    uint16_t table_select[3]; /*!< 10bit(window_switching_flag == 1), 15bit(window_switching_flag == 0) big0_band, big1_band, big2_bandのハフマン符号化テーブルインデックス */
    uint16_t subblock_gain[3]; /*!< 9bit 量子化ステップで使用（dist10では常に0） */
    uint8_t region0_count; /*!< 4bit(window_switching_flag == 0) big1_bandの最小の周波数を定めるスケールファクタのバンドインデックス */
    uint8_t region1_count; /*!< 3bit(window_switching_flag == 0) big2_bandの最小の周波数を定めるスケールファクタのバンドインデックス */
    uint8_t preflag; /*!< 1bit プリエンファシスで増幅されたら1, そうでなければ0 */
    uint8_t scalefac_scale; /*!< 1bit プリエンファシス, amp_scalefac_bandsで使われる値（dist10では常に0） */
    uint8_t count1table_select; /*!< 1bit count1_bandのハフマン符号化テーブルインデックス */
};

/*! 付加情報（サイドインフォメーション） */
struct MP3SideInformation {
    uint16_t main_data_begin; /*!< 9bit メインデータが始まるまでの負のオフセットバイト数 */
    uint8_t private_bits; /*!< ステレオであれば3bit, モノラルであれば5bit ユーザ向けのビット(ISOは未使用)  */
    struct {
        uint8_t scfsi[4]; /*!< 1bitx4 ScaleFector Selection Information 4グループ(0-5,6-10,11-15,16-20)で、2グラニュールで同一のスケールファクタを使用しているか？  */
        struct MP3GranuleInformation gr[2]; /*!< グラニュール情報（1フレーム2グラニュール） */
    } ch[MP3_MAX_NUM_CHANNELS];
};

/*! フォーマット情報 */
struct DGFormatInformation {
    int32_t num_channels; /*!< チャンネル数 */
    int32_t num_samples; /*!< サンプル数 */
    int32_t sampling_rate; /*!< サンプリングレート */
    int32_t bitrate; /*!< ビットレート */
};

/*! デコーダハンドル */
struct DGDecoder;

#ifdef __cplusplus
extern "C" {
#endif

/*! ハンドル生成に必要なワークサイズ計算 */
int32_t DGDecoder_CalculateWorkSize(void);

/*! ハンドル生成 */
struct DGDecoder* DGDecoder_Create(void *work, int32_t work_size);

/*! ハンドル破棄 */
void DGDecoder_Destroy(struct DGDecoder *decoder);

/*! ハンドルのリセット */
void DGDecoder_Reset(struct DGDecoder *decoder);

/*! 1フレームデコード */
enum DGDecoderDecodeResult DGDecoder_DecodeFrame(
    struct DGDecoder *decoder, const uint8_t *data, size_t data_size,
    struct MP3FrameHeader *header, struct MP3SideInformation *side_info,
    int16_t **decoded_buffer, int32_t buffer_num_channels, int32_t buffer_num_samples,
    size_t *decoded_size);

/*! 全データフレームデコード */
enum DGDecoderDecodeResult DGDecoder_DecodeWhole(
    struct DGDecoder *decoder, const uint8_t *data, size_t data_size,
    int16_t **buffer, int32_t num_buffer_channels, int32_t num_buffer_samples,
    int32_t *num_decoded_samples);

/*! フォーマット情報の取得 */
enum DGDecoderDecodeResult DGDecoder_GetFormatInformation(
    const uint8_t *data, size_t data_size, struct DGFormatInformation *format);

#ifdef __cplusplus
}
#endif

#endif /* DGDECODER_H_INCLUDED */

