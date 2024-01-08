#ifndef DGHUFFMAN_H_INCLUDED
#define DGHUFFMAN_H_INCLUDED

#include <stdint.h>

#include "dg_maindata_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* bigvalueの復号 */
void DGHuffman_DecodeBigValue(
    int32_t table_index, struct DGMainDataBuffer *buffer, int32_t *x, int32_t *y);

/*! count1_dataの復号 */
void DGHuffman_DecodeCount1Data(
    int32_t table_index, struct DGMainDataBuffer *buffer, int32_t *x, int32_t *y, int32_t *v, int32_t *w);

#ifdef __cplusplus
}
#endif

#endif /* DGHUFFMAN_H_INCLUDED */
