#define PY_SSIZE_T_CLEAN
#include "split_dtype_functions.h"
#include <Python.h>
#include <stdint.h>
#include <time.h>

// Helper function that count zero bytes
static void count_zero_bytes(const uint8_t *src, Py_ssize_t len,
                             Py_ssize_t *msb_zeros, Py_ssize_t *mid_high,
                             Py_ssize_t *mid_low, Py_ssize_t *lsb_zeros);

///////////////////////////////////
/// Split Helper Functions ///////
//////////////////////////////////

uint32_t reorder_float_bits_dtype32(float number);
// void reorder_all_floats_dtype32(uint8_t *src, Py_ssize_t len);static int
// allocate_4chunk_buffs(uint8_t **buf1, uint8_t **buf2, uint8_t **buf3,
//                              uint8_t **buf4, Py_ssize_t size1, Py_ssize_t
//                              size2,
//                            Py_ssize_t size3, Py_ssize_t size4);

int split_bytearray_dtype32(uint8_t *src, Py_ssize_t len,
                            uint8_t **chunk_buffs, size_t *bufLens,
                            int bits_mode, int bytes_mode, int is_review,
                            int threads);

///////////////////////////////////
/////////  Combine Functions //////
///////////////////////////////////

// Reordering function for float bits
uint32_t revert_float_bits(float number);
void revert_all_floats(uint8_t *src, Py_ssize_t len);
int allocate_buffer(uint8_t **result, Py_ssize_t total_len);

uint8_t combine_buffers_dtype32(uint8_t *buf1, uint8_t *buf2, uint8_t *buf3,
                                 uint8_t *buf4, uint8_t *combinePtr,
                                 Py_ssize_t *bufLens, int bits_mode,
                                 int bytes_mode, int threads);

// Helper function
int buffer_ratio_dtype32(int bytes_mode, uint32_t *buf_ratio);
