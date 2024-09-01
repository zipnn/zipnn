#ifndef DATA_MANIPULATION_DTYPE16_H 
#define DATA_MANIPULATION_DTYPE16_H

// Reordering function for float bits
static uint32_t reorder_float_bits_dtype16(float number); 
static void reorder_all_floats_dtype16(u_int8_t *src, Py_ssize_t len);
int split_bytearray_dtype16(u_int8_t *src, Py_ssize_t len, u_int8_t **buffers,
                           int bits_mode, int bytes_mode, int is_review,
                           int threads);

static uint32_t revert_float_bits_dtype16(float number); 
static void revert_all_floats_dtype16(u_int8_t *src, Py_ssize_t len); 
int combine_buffers_dtype16(u_int8_t *buf1, u_int8_t *buf2, u_int8_t *combinePtr,
                           Py_ssize_t half_len, int bits_mode, int bytes_mode,
                           int threads);
#endif // DATA_MANIPULATION_DTYPE16_H
