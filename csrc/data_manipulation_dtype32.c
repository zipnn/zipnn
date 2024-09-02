#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <time.h>
#include <data_manipulation_dtype32.h>

// Helper function that count zero bytes
static void count_zero_bytes(const u_int8_t *src, Py_ssize_t len,
                             Py_ssize_t *msb_zeros, Py_ssize_t *mid_high,
                             Py_ssize_t *mid_low, Py_ssize_t *lsb_zeros) {
  Py_ssize_t num_uint32 =
      len /
      sizeof(
          uint32_t);  // Calculate how many uint32_t elements are in the buffer
  uint32_t *uint32_array =
      (uint32_t *)src;  // Cast the byte buffer to a uint32_t array

  *msb_zeros = 0;
  *mid_high = 0;
  *mid_low = 0;
  *lsb_zeros = 0;

  for (Py_ssize_t i = 0; i < num_uint32; i++) {
    uint32_t value = uint32_array[i];
    if (((value >> 24) & 0xFF) == 0)
      (*msb_zeros)++;
    if (((value >> 16) & 0xFF) == 0)
      (*mid_high)++;
    if (((value >> 8) & 0xFF00) == 0)
      (*mid_low)++;
    if ((value & 0xFF) == 0)
      (*lsb_zeros)++;
  }
}

///////////////////////////////////
/// Split Helper Functions ///////
//////////////////////////////////

// Reordering function for float bits
uint32_t reorder_float_bits(float number) {
  union {
    float f;
    uint32_t u;
  } value = {.f = number};

  uint32_t sign = (value.u >> 8) & 0x800000;
  uint32_t exponent = (value.u << 1) & 0xFF000000;
  uint32_t mantissa = (value.u) & 0x7FFFFF;
  return exponent | sign | mantissa;
}

// Helper function to reorder all floats in a bytearray
void reorder_all_floats(u_int8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = reorder_float_bits(*(float *)&uint_array[i]);
  }
}

int allocate_4buffers(u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                             u_int8_t **buf4, Py_ssize_t size1, Py_ssize_t size2,
                             Py_ssize_t size3, Py_ssize_t size4) {
  *buf1 = (size1 > 0) ? PyMem_Malloc(size1) : NULL;
  *buf2 = (size2 > 0) ? PyMem_Malloc(size2) : NULL;
  *buf3 = (size3 > 0) ? PyMem_Malloc(size3) : NULL;
  *buf4 = (size4 > 0) ? PyMem_Malloc(size4) : NULL;

  if ((size1 > 0 && *buf1 == NULL) || (size2 > 0 && *buf2 == NULL) ||
      (size3 > 0 && *buf3 == NULL) || (size4 > 0 && *buf4 == NULL)) {
    PyMem_Free(*buf1);
    PyMem_Free(*buf2);
    PyMem_Free(*buf3);
    PyMem_Free(*buf4);
    return -1;
  }
  return 0;
}

int handle_split_mode_220(u_int8_t *src, Py_ssize_t total_len,
                                 u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                                 u_int8_t **buf4, Py_ssize_t *buf1_len,
                                 Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                                 Py_ssize_t *buf4_len) {
  Py_ssize_t q_len = total_len / 4;
  *buf1_len = q_len;
  *buf2_len = q_len;
  *buf3_len = q_len;
  *buf4_len = q_len;

  if (allocate_4buffers(buf1, buf2, buf3, buf4, *buf1_len, *buf2_len, *buf3_len,
                        *buf4_len) != 0)
    return -1;

  u_int8_t *dst1 = *buf1, *dst2 = *buf2, *dst3 = *buf3, *dst4 = *buf4;

  for (Py_ssize_t i = 0; i < total_len; i += 4) {
    *dst1++ = src[i];
    *dst2++ = src[i + 1];
    *dst3++ = src[i + 2];
    *dst4++ = src[i + 3];
  }
  return 0;
}

int handle_split_mode_41(u_int8_t *src, Py_ssize_t total_len,
                                u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                                u_int8_t **buf4, Py_ssize_t *buf1_len,
                                Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                                Py_ssize_t *buf4_len) {
  Py_ssize_t three_q_len = total_len / 4 * 3;
  *buf1_len = three_q_len;
  *buf2_len = 0;
  *buf3_len = 0;
  *buf4_len = 0;

  if (allocate_4buffers(buf1, buf2, buf3, buf4, *buf1_len, *buf2_len, *buf3_len,
                        *buf4_len) != 0)
    return -1;

  u_int8_t *dst1 = *buf1;

  Py_ssize_t j = 0;
  for (Py_ssize_t i = 0; i < total_len; i += 4) {
    dst1[j++] = src[i];
    dst1[j++] = src[i + 1];
    dst1[j++] = src[i + 2];
  }
  return 0;
}

int handle_split_mode_9(u_int8_t *src, Py_ssize_t total_len,
                               u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                               u_int8_t **buf4, Py_ssize_t *buf1_len,
                               Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                               Py_ssize_t *buf4_len) {
  Py_ssize_t half_len = total_len / 2;
  *buf1_len = half_len;
  *buf2_len = 0;
  *buf3_len = 0;
  *buf4_len = 0;

  if (allocate_4buffers(buf1, buf2, buf3, buf4, *buf1_len, *buf2_len, *buf3_len,
                        *buf4_len) != 0)
    return -1;

  uint32_t *src_uint32 = (uint32_t *)src;
  uint16_t *dst_uint16 = (uint16_t *)*buf1;

  Py_ssize_t num_uint32 = total_len / sizeof(uint32_t);

  for (Py_ssize_t i = 0; i < num_uint32; i++) {
    dst_uint16[i] = (uint16_t)src_uint32[i];
  }

  return 0;
}

static int handle_split_mode_1(u_int8_t *src, Py_ssize_t total_len,
                               u_int8_t **buf1, u_int8_t **buf2, u_int8_t **buf3,
                               u_int8_t **buf4, Py_ssize_t *buf1_len,
                               Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
                               Py_ssize_t *buf4_len) {
  Py_ssize_t half_len = total_len / 4;
  *buf1_len = half_len;
  *buf2_len = 0;
  *buf3_len = 0;
  *buf4_len = 0;

  if (allocate_4buffers(buf1, buf2, buf3, buf4, *buf1_len, *buf2_len, *buf3_len,
                        *buf4_len) != 0)
    return -1;

  uint32_t *src_uint32 = (uint32_t *)src;
  u_int8_t *dst_uint8 = (u_int8_t *)*buf1;

  Py_ssize_t num_uint32 = total_len / sizeof(uint32_t);

  for (Py_ssize_t i = 0; i < num_uint32; i++) {
    dst_uint8[i] = (u_int8_t)src_uint32[i];
  }

  return 0;
}

// Helper function to split a bytearray into four buffers
int split_dtype32(u_int8_t *src, Py_ssize_t total_len, u_int8_t **buf1,
                         u_int8_t **buf2, u_int8_t **buf3, u_int8_t **buf4,
                         Py_ssize_t *buf1_len, Py_ssize_t *buf2_len,
                         Py_ssize_t *buf3_len, Py_ssize_t *buf4_len,
                         int bits_mode, int bytes_mode, int is_review,
                         int threads) {
  if (bits_mode == 1) {  // reoreder exponent
    reorder_all_floats(src, total_len);
  }

  if (is_review == 1) {
    clock_t start, end;
    double cpu_time_used;
    Py_ssize_t msb_zeros, mid_high, mid_low, low;
    start = clock();
    count_zero_bytes(src, total_len, &msb_zeros, &mid_high, &mid_low, &low);

    end = clock();  // End the timer
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("msb_zeros %zd mid_high %zd mid_low %zd low %zd time in sec %f \n",
           msb_zeros, mid_high, mid_low, low, cpu_time_used);
  }

  switch (bytes_mode) {
  case 220:
    // 8b1_10_11_100 [decimal 220] - bytegroup to four groups [1,2,3,4]
    handle_split_mode_220(src, total_len, buf1, buf2, buf3, buf4, buf1_len,
                          buf2_len, buf3_len, buf4_len);
    break;

  case 41:
    // 8b0_01_01_001 [decimal 41] - truncate the MSB [0,1,1,1]
    handle_split_mode_41(src, total_len, buf1, buf2, buf3, buf4, buf1_len,
                         buf2_len, buf3_len, buf4_len);
    break;

  case 9:
    //     8b0_00_01_001 [decimal 9] - truncate the MSB+MID_HIGH [0,0,1,1]
    handle_split_mode_9(src, total_len, buf1, buf2, buf3, buf4, buf1_len,
                        buf2_len, buf3_len, buf4_len);
    break;

  case 1:
    //     8b0_00_00_001 [decimal 1] - truncate the MSB+MID_HIGH+MID_LOW
    //     [0,0,0,1]
    handle_split_mode_1(src, total_len, buf1, buf2, buf3, buf4, buf1_len,
                        buf2_len, buf3_len, buf4_len);
    break;

  default:
    // we are not supportin this splitting bytes_mode
    return -1;
  }
  return 0;
}

///////////////////////////////////
/////////  Combine Functions //////
///////////////////////////////////

// Reordering function for float bits
static uint32_t revert_float_bits(float number) {
  union {
    float f;
    uint32_t u;
  } value = {.f = number};

  uint32_t sign = (value.u << 8) & 0x80000000;
  uint32_t exponent = (value.u >> 1) & 0x7F800000;
  uint32_t mantissa = (value.u) & 0x7FFFFF;
  return sign | exponent | mantissa;
}

// Helper function to reorder all floats in a bytearray
void revert_all_floats(u_int8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = revert_float_bits(*(float *)&uint_array[i]);
  }
}

static int allocate_buffer(u_int8_t **result, Py_ssize_t total_len) {
  *result = PyMem_Malloc(total_len);
  if (*result == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    return -1;
  }
  return 0;
}

static int handle_combine_mode_220(u_int8_t **result, Py_ssize_t *total_len,
                                   u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                   u_int8_t *buf4, Py_ssize_t buf1_len,
                                   Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                   Py_ssize_t buf4_len) {
  *total_len = buf1_len * 4;
  if (allocate_buffer(result, *total_len) != 0)
    return -1;

  u_int8_t *dst = *result;
  Py_ssize_t q_len = *total_len / 4;

  Py_ssize_t j = 0;
  for (Py_ssize_t i = 0; i < q_len; i++) {
    dst[j++] = buf1[i];
    dst[j++] = buf2[i];
    dst[j++] = buf3[i];
    dst[j++] = buf4[i];
  }
  return 0;
}

static int handle_combine_mode_41(u_int8_t **result, Py_ssize_t *total_len,
                                  u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                  u_int8_t *buf4, Py_ssize_t buf1_len,
                                  Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                  Py_ssize_t buf4_len) {
  *total_len = buf1_len / 3 * 4;
  if (allocate_buffer(result, *total_len) != 0)
    return -1;

  u_int8_t *dst = *result;

  Py_ssize_t i = 0;
  for (Py_ssize_t j = 0; j < *total_len; j += 4) {
    dst[j] = buf1[i++];
    dst[j + 1] = buf1[i++];
    dst[j + 2] = buf1[i++];
    dst[j + 3] = 0;
  }
  return 0;
}

static int handle_combine_mode_9(u_int8_t **result, Py_ssize_t *total_len,
                                 u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                 u_int8_t *buf4, Py_ssize_t buf1_len,
                                 Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                 Py_ssize_t buf4_len) {
  *total_len = buf1_len * 2;
  if (allocate_buffer(result, *total_len) != 0)
    return -1;

  uint32_t *dst_uint32 = (uint32_t *)*result;
  uint16_t *src_uint16 = (uint16_t *)buf1;

  Py_ssize_t num_uint32 = *total_len / sizeof(uint32_t);

  for (Py_ssize_t i = 0; i < num_uint32; i++) {
    dst_uint32[i] = src_uint16[i];
  }

  return 0;
}

static int handle_combine_mode_1(u_int8_t **result, Py_ssize_t *total_len,
                                 u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                 u_int8_t *buf4, Py_ssize_t buf1_len,
                                 Py_ssize_t buf2_len, Py_ssize_t buf3_len,
                                 Py_ssize_t buf4_len) {
  *total_len = buf1_len * 4;
  if (allocate_buffer(result, *total_len) != 0)
    return -1;

  uint32_t *dst_uint32 = (uint32_t *)*result;
  u_int8_t *src_uint8 = (u_int8_t *)buf1;

  Py_ssize_t num_uint32 = *total_len / sizeof(uint32_t);

  for (Py_ssize_t i = 0; i < num_uint32; i++) {
    dst_uint32[i] = src_uint8[i];
  }

  return 0;
}

// Helper function to combine four buffers into a single bytearray
u_int8_t *combine_dtype32(Py_ssize_t *total_len, u_int8_t *buf1,
                                u_int8_t *buf2, u_int8_t *buf3, u_int8_t *buf4,
                                Py_ssize_t buf1_len, Py_ssize_t buf2_len,
                                Py_ssize_t buf3_len, Py_ssize_t buf4_len,
                                int bytes_mode, int threads) {
  u_int8_t *result = NULL;

  switch (bytes_mode) {
  case 220:
    // 8b1_10_11_100 [decimal 220] - bytegroup to four groups [1,2,3,4]
    handle_combine_mode_220(&result, total_len, buf1, buf2, buf3, buf4,
                            buf1_len, buf2_len, buf3_len, buf4_len);
    break;

  case 41:
    // 8b0_01_01_001 [decimal 41] - truncate the MSB [0,1,1,1]
    handle_combine_mode_41(&result, total_len, buf1, buf2, buf3, buf4, buf1_len,
                           buf2_len, buf3_len, buf4_len);
    break;

  case 9:
    //     8b0_00_01_001 [decimal 9] - truncate the MSB+MID_HIGH [0,0,1,1]
    handle_combine_mode_9(&result, total_len, buf1, buf2, buf3, buf4, buf1_len,
                          buf2_len, buf3_len, buf4_len);
    break;

  case 1:
    //     8b0_00_00_001 [decimal 1] - truncate the MSB+MID_HIGH+MID_LOW
    //     [0,0,0,1]
    handle_combine_mode_1(&result, total_len, buf1, buf2, buf3, buf4, buf1_len,
                          buf2_len, buf3_len, buf4_len);
    break;

  default:
    // we are not supportin this splitting bytes_mode
    return NULL;
  }
  return result;
}
