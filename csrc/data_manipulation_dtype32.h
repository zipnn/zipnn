#define PY_SSIZE_T_CLEAN
#include "split_dtype_functions.h"
#include <Python.h>
#include <stdint.h>
#include <time.h>

// Helper function that count zero bytes
static void count_zero_bytes(const u_int8_t *src, Py_ssize_t len,
                             Py_ssize_t *msb_zeros, Py_ssize_t *mid_high,
                             Py_ssize_t *mid_low, Py_ssize_t *lsb_zeros); 

///////////////////////////////////
/// Split Helper Functions ///////
//////////////////////////////////

uint32_t reorder_float_bits(float number);
void reorder_all_floats(u_int8_t *src, Py_ssize_t len);static int allocate_4buffers(u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                             u_int8_t **buf4, Py_ssize_t size1, Py_ssize_t size2,
                             Py_ssize_t size3, Py_ssize_t size4);
int handle_split_mode_220(u_int8_t *src, Py_ssize_t total_len,
                                 u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                                 u_int8_t **buf4, Py_ssize_t *buf1_len,
                                 Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                                 Py_ssize_t *buf4_len);static int handle_split_mode_41(u_int8_t *src, Py_ssize_t total_len,
                                u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                                u_int8_t **buf4, Py_ssize_t *buf1_len,
                                Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                                Py_ssize_t *buf4_len);
int handle_split_mode_9(u_int8_t *src, Py_ssize_t total_len,
                               u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                               u_int8_t **buf4, Py_ssize_t *buf1_len,
                               Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                               Py_ssize_t *buf4_len);static int handle_split_mode_1(u_int8_t *src, Py_ssize_t total_len,
                               u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                               u_int8_t **buf4, Py_ssize_t *buf1_len,
                               Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                               Py_ssize_t *buf4_len);
int split_dtype32(u_int8_t *src, Py_ssize_t total_len, u_int8_t **buf1,
                         u_int8_t **buf2, u_int8_t **buf3, u_int8_t **buf4,
                         Py_ssize_t *buf1_len, Py_ssize_t *buf2_len,
                         Py_ssize_t *buf3_len, Py_ssize_t *buf4_len,
                         int bits_mode, int bytes_mode, int is_review,
                         int threads);

///////////////////////////////////
/////////  Combine Functions //////
///////////////////////////////////

// Reordering function for float bits
static uint32_t revert_float_bits(float number); 
void revert_all_floats(u_int8_t *src, Py_ssize_t len);
static int allocate_buffer(u_int8_t **result, Py_ssize_t total_len);
static int handle_combine_mode_220(u_int8_t **result, Py_ssize_t *total_len,
                                   u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                   u_int8_t *buf4, Py_ssize_t buf1_len,
                                   Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                   Py_ssize_t buf4_len);
static int handle_combine_mode_41(u_int8_t **result, Py_ssize_t *total_len,
                                  u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                  u_int8_t *buf4, Py_ssize_t buf1_len,
                                  Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                  Py_ssize_t buf4_len); 
static int handle_combine_mode_9(u_int8_t **result, Py_ssize_t *total_len,
                                 u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                 u_int8_t *buf4, Py_ssize_t buf1_len,
                                 Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                 Py_ssize_t buf4_len);
static int handle_combine_mode_1(u_int8_t **result, Py_ssize_t *total_len,
                                 u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                 u_int8_t *buf4, Py_ssize_t buf1_len,
                                 Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                 Py_ssize_t buf4_len);
u_int8_t *combine_dtype32(Py_ssize_t *total_len, u_int8_t *buf1,
                                u_int8_t *buf2, u_int8_t *buf3, u_int8_t *buf4,
                                Py_ssize_t buf1_len, Py_ssize_t buf2_len,
                                Py_ssize_t buf3_len, Py_ssize_t buf4_len,
                                int bytes_mode, int threads);
