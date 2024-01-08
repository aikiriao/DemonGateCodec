#include "dg_maindata_buffer.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

/*! バッファの内部状態クリア */
void DGMainDataBuffer_Reset(struct DGMainDataBuffer *buffer)
{
    /* メインデータバッファの初期化 */
    memset(buffer->maindata_buffer, 0, sizeof(buffer->maindata_buffer));
    BitReader_Open(&(buffer->reader), &(buffer->maindata_buffer[0]), sizeof(buffer->maindata_buffer));
    buffer->buffer_write_pos = 0;
}

/*! 読み込んだビット数を返す */
int32_t DGMainDataBuffer_GetTotalReadBits(const struct DGMainDataBuffer *buffer)
{
    int32_t bits;

    assert(buffer != NULL);

    /* バイト単位で取得し、ビット単位に換算 */
    BitStream_Tell((struct BitStream *)&buffer->reader, &bits);
    bits *= 8;

    /* バッファリングしている分を減算 */
    return bits - (int32_t)buffer->reader.bit_count;
}

/*! バッファにデータ書き出し */
void DGMainDataBuffer_PutDataToBuffer(struct DGMainDataBuffer *buffer, const uint8_t *data, size_t data_size)
{
    /* バッファから飛び出る場合は、末尾までまず書き込む */
    if ((buffer->buffer_write_pos + data_size) > MP3_BUFFER_SIZE) {
        const size_t tail_size = MP3_BUFFER_SIZE - buffer->buffer_write_pos;
        memcpy(&buffer->maindata_buffer[buffer->buffer_write_pos], data, tail_size);
        data_size -= tail_size;
        data += tail_size;
        buffer->buffer_write_pos = 0;
    }

    /* バッファに書き込み */
    memcpy(&buffer->maindata_buffer[buffer->buffer_write_pos], data, data_size);
    data += data_size;
    buffer->buffer_write_pos += (int32_t)data_size;
}

/*! バッファからビット読み出し */
uint32_t DGMainDataBuffer_GetBitsFromBuffer(struct DGMainDataBuffer *buffer, uint32_t nbits)
{
    uint32_t buf;
    struct BitStream *reader;

    assert(buffer != NULL);
    assert(nbits <= 32);

    reader = &buffer->reader;

    /* バッファ終端を超えて読もうとしたときは末尾まで読んで先頭にシーク */
    if (reader->memory_p >= reader->memory_tail) {
        const int32_t total_read_bits = DGMainDataBuffer_GetTotalReadBits(buffer);
        const uint32_t remain_bits = (uint32_t)(8 * reader->memory_size) - (uint32_t)total_read_bits;
        if (nbits > remain_bits) {
            uint32_t tail;
            BitReader_GetBits(reader, &tail, remain_bits);
            BitStream_Seek(reader, 0, BITSTREAM_SEEK_SET);
            BitReader_GetBits(reader, &buf, nbits - remain_bits);
            return (tail << (nbits - remain_bits)) | buf;
        }
    }

    BitReader_GetBits(reader, &buf, nbits);

    /* バッファの終端を超えて読んでしまった場合は先頭に戻りつつ訂正 */
    if (reader->memory_p > reader->memory_tail) {
        const int32_t total_read_bits = DGMainDataBuffer_GetTotalReadBits(buffer);
        if (total_read_bits >= (int32_t)(8 * reader->memory_size)) {
            uint32_t head;
            const uint32_t overrun_bits = (uint32_t)total_read_bits - (uint32_t)(8 * reader->memory_size);
            const uint32_t overrun_mask = (1U << overrun_bits) - 1;
            BitStream_Seek(reader, 0, BITSTREAM_SEEK_SET);
            BitReader_GetBits(reader, &head, overrun_bits);
            return (buf & ~overrun_mask) | (head & overrun_mask);
        }
    }

    return buf;
}
