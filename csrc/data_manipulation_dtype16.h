#ifndef DATA_MANIPULATION_DTYPE16_H
#define DATA_MANIPULATION_DTYPE16_H

// Reordering function for float bits
static uint32_t reorder_float_bits_dtype16(float number);
static void reorder_all_floats_dtype16(uint8_t *src, size_t len);
int split_bytearray_dtype16(uint8_t *src, size_t len, uint8_t **chunk_buffs,
                            size_t *unCompChunksSizeCurChunk, int bits_mode,
                            int bytes_mode, int is_review);

static uint32_t revert_float_bits_dtype16(float number);
static void revert_all_floats_dtype16(uint8_t *src, size_t len);
int combine_buffers_dtype16(uint8_t *buf1, uint8_t *buf2, uint8_t *combinePtr,
                            size_t *bufLens, int bits_mode, int bytes_mode);

int buffer_ratio_dtype16(int bytes_mode, uint32_t *buf_ratio);

#endif // DATA_MANIPULATION_DTYPE16_H
