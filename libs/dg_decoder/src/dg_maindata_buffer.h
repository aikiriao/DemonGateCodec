#ifndef DGMAINDATABUFFER_H_INCLUDED
#define DGMAINDATABUFFER_H_INCLUDED

#include <stdint.h>

#include "bit_stream.h"

#define MP3_BUFFER_SIZE 4096 /*!< バッファサイズ */

struct DGMainDataBuffer {
    uint8_t maindata_buffer[MP3_BUFFER_SIZE];
    struct BitStream reader;
    int32_t buffer_write_pos;
};

#ifdef __cplusplus
extern "C" {
#endif

/*! バッファの内部状態クリア */
void DGMainDataBuffer_Reset(struct DGMainDataBuffer *buffer);

/*! 読み込んだビット数を返す */
int32_t DGMainDataBuffer_GetTotalReadBits(const struct DGMainDataBuffer *buffer);

/*! バッファにデータ書き出し */
void DGMainDataBuffer_PutDataToBuffer(struct DGMainDataBuffer *buffer, const uint8_t *data, size_t data_size);

/*! バッファからビット読み出し */
uint32_t DGMainDataBuffer_GetBitsFromBuffer(struct DGMainDataBuffer *buffer, uint32_t nbits);

#ifdef __cplusplus
}
#endif

#endif /* DGMAINDATABUFFER_H_INCLUDED */

