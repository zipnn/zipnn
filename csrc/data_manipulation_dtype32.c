#include <Python.h>
#include <stdint.h>
#include <time.h>

//// Helper function that count zero bytes
static void count_zero_bytes(const uint8_t *src, size_t len, size_t *msb_zeros,
                             size_t *mid_high, size_t *mid_low,
                             size_t *lsb_zeros) {
  size_t num_uint32 =
      len /
      sizeof(
          uint32_t); // Calculate how many uint32_t elements are in the buffer
  const uint32_t *uint32_array =
      (uint32_t *)src; // Cast the byte buffer to a uint32_t array

  *msb_zeros = 0;
  *mid_high = 0;
  *mid_low = 0;
  *lsb_zeros = 0;

  for (size_t i = 0; i < num_uint32; i++) {
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
//
/////////////////////////////////////
///// Split Helper Functions ///////
////////////////////////////////////
//
//// Reordering function for float bits
uint32_t reorder_float_bits_dtype32(float number) {
  union {
    float f;
    uint32_t u;
  } value = {.f = number};

  uint32_t sign = (value.u >> 8) & 0x800000;
  uint32_t exponent = (value.u << 1) & 0xFF000000;
  uint32_t mantissa = (value.u) & 0x7FFFFF;
  return exponent | sign | mantissa;
}
//
//// Helper function to reorder all floats in a bytearray
void reorder_all_floats_dtype32(uint8_t *src, size_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  size_t num_floats = len / sizeof(uint32_t);
  for (size_t i = 0; i < num_floats; i++) {
    uint_array[i] = reorder_float_bits_dtype32(*(float *)&uint_array[i]);
  }
}

//
int allocate_4chunk_buffs(uint8_t **chunk_buffs, const size_t *bufLens) {
  chunk_buffs[0] = (bufLens[0] > 0) ? malloc(bufLens[0]) : NULL;
  chunk_buffs[1] = (bufLens[1] > 0) ? malloc(bufLens[1]) : NULL;
  chunk_buffs[2] = (bufLens[2] > 0) ? malloc(bufLens[2]) : NULL;
  chunk_buffs[3] = (bufLens[3] > 0) ? malloc(bufLens[3]) : NULL;

  if ((bufLens[0] > 0 && chunk_buffs[0] == NULL) ||
      (bufLens[1] > 0 && chunk_buffs[1] == NULL) ||
      (bufLens[2] > 0 && chunk_buffs[2] == NULL) ||
      (bufLens[3] > 0 && chunk_buffs[3] == NULL)) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory, allocate 4 buffs");
    return -1;
  }
  return 0;
}

int handle_split_mode_220(const uint8_t *src, size_t total_len,
                          uint8_t **chunk_buffs, size_t *bufLens,
                          uint32_t num_buf) {
  size_t q_len = total_len / num_buf;
  uint32_t remainder = total_len % num_buf;

  for (uint32_t b = 0; b < num_buf; b++) {
    if (b < remainder) {
      bufLens[b] = q_len + 1;
    } else {
      bufLens[b] = q_len;
    }
  }

  if (allocate_4chunk_buffs(chunk_buffs, bufLens) != 0)
    return -1;

  uint8_t *dst1 = chunk_buffs[0], *dst2 = chunk_buffs[1],
          *dst3 = chunk_buffs[2], *dst4 = chunk_buffs[3];

  for (size_t i = 0; i < total_len; i += 4) {
    *dst1++ = src[i];
    *dst2++ = src[i + 1];
    *dst3++ = src[i + 2];
    *dst4++ = src[i + 3];
  }

  switch (remainder) {
  case 3:
    *dst3++ = src[total_len - 2];
  case 2:
    *dst2++ = src[total_len - 1];
  case 1:
    *dst1++ = src[total_len - remainder];
    break;
  default:
    break;
  }

  if (remainder == 3) {
    *dst1++ = src[total_len - remainder];
    *dst2++ = src[total_len - remainder - 1];
    *dst3++ = src[total_len - remainder - 2];
  }

  if (remainder == 2) {
    *dst1++ = src[total_len - remainder];
    *dst2++ = src[total_len - remainder - 1];
  }

  if (remainder == 1) {
    *dst1++ = src[total_len - remainder];
  }

  return 0;
}
//
// int handle_split_mode_41(uint8_t *src, size_t total_len,
//                                uint8_t **buf1, uint8_t **buf2, uint8_t
//                                **buf3, uint8_t **buf4, size_t *buf1_len,
//                                size_t *buf2_len, size_t *buf3_len,
//                                size_t *buf4_len) {
//  size_t three_q_len = total_len / 4 * 3;
//  *buf1_len = three_q_len;
//  *buf2_len = 0;
//  *buf3_len = 0;
//  *buf4_len = 0;
//
//  if (allocate_4chunk_buffs(buf1, buf2, buf3, buf4, *buf1_len, *buf2_len,
//  *buf3_len,
//                        *buf4_len) != 0)
//    return -1;
//
//  uint8_t *dst1 = *buf1;
//
//  size_t j = 0;
//  for (size_t i = 0; i < total_len; i += 4) {
//    dst1[j++] = src[i];
//    dst1[j++] = src[i + 1];
//    dst1[j++] = src[i + 2];
//  }
//  return 0;
//}
//
// int handle_split_mode_9(uint8_t *src, size_t total_len,
//                               uint8_t **buf1, uint8_t **buf2, uint8_t
//                               **buf3, uint8_t **buf4, size_t *buf1_len,
//                               size_t *buf2_len, size_t *buf3_len,
//                               size_t *buf4_len) {
//  size_t half_len = total_len / 2;
//  *buf1_len = half_len;
//  *buf2_len = 0;
//  *buf3_len = 0;
//  *buf4_len = 0;
//
//  if (allocate_4chunk_buffs(buf1, buf2, buf3, buf4, *buf1_len, *buf2_len,
//  *buf3_len,
//                        *buf4_len) != 0)
//    return -1;
//
//  uint32_t *src_uint32 = (uint32_t *)src;
//  uint16_t *dst_uint16 = (uint16_t *)*buf1;
//
//  size_t num_uint32 = total_len / sizeof(uint32_t);
//
//  for (size_t i = 0; i < num_uint32; i++) {
//    dst_uint16[i] = (uint16_t)src_uint32[i];
//  }
//
//  return 0;
//}
//
// static int handle_split_mode_1(uint8_t *src, size_t total_len,
//                               uint8_t **buf1, uint8_t **buf2, uint8_t
//                               **buf3, uint8_t **buf4, size_t *buf1_len,
//                               size_t *buf2_len, size_t *buf3_len,
//                               size_t *buf4_len) {
//  size_t half_len = total_len / 4;
//  *buf1_len = half_len;
//  *buf2_len = 0;
//  *buf3_len = 0;
//  *buf4_len = 0;
//
//  if (allocate_4chunk_buffs(buf1, buf2, buf3, buf4, *buf1_len, *buf2_len,
//  *buf3_len,
//                        *buf4_len) != 0)
//    return -1;
//
//  uint32_t *src_uint32 = (uint32_t *)src;
//  uint8_t *dst_uint8 = (uint8_t *)*buf1;
//
//  size_t num_uint32 = total_len / sizeof(uint32_t);
//
//  for (size_t i = 0; i < num_uint32; i++) {
//    dst_uint8[i] = (uint8_t)src_uint32[i];
//  }
//
//  return 0;
//}
//
//// Helper function to split a bytearray into four chunk_buffs
int split_bytearray_dtype32(uint8_t *src, size_t len, uint8_t **chunk_buffs,
                            size_t *bufLens, int bits_mode, int bytes_mode,
                            int is_review) {
  uint32_t num_buf = 4;
  if (bits_mode == 1) { // reoreder exponent
    reorder_all_floats_dtype32(src, len);
  }

  if (is_review == 1) {
    //    clock_t start, end;
    //    double cpu_time_used;
    size_t msb_zeros, mid_high, mid_low, low;
    //    start = clock();
    count_zero_bytes(src, len, &msb_zeros, &mid_high, &mid_low, &low);

    //    end = clock();  // End the timer
    //    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  }

  switch (bytes_mode) {
  case 220:
    // 8b1_10_11_100 [decimal 220] - bytegroup to four groups [1,2,3,4]
    handle_split_mode_220(src, len, chunk_buffs, bufLens, num_buf);
    break;
    //
    //  case 41:
    //    // 8b0_01_01_001 [decimal 41] - truncate the MSB [0,1,1,1]
    //    handle_split_mode_41(src, total_len, buf1, buf2, buf3, buf4, buf1_len,
    //                         buf2_len, buf3_len, buf4_len);
    //    break;
    //
    //  case 9:
    //    //     8b0_00_01_001 [decimal 9] - truncate the MSB+MID_HIGH [0,0,1,1]
    //    handle_split_mode_9(src, total_len, buf1, buf2, buf3, buf4, buf1_len,
    //                        buf2_len, buf3_len, buf4_len);
    //    break;
    //
    //  case 1:
    //    //     8b0_00_00_001 [decimal 1] - truncate the MSB+MID_HIGH+MID_LOW
    //    //     [0,0,0,1]
    //    handle_split_mode_1(src, total_len, buf1, buf2, buf3, buf4, buf1_len,
    //                        buf2_len, buf3_len, buf4_len);
    //    break;
    //
  default:
    // we are not supportin this splitting bytes_mode
    return -1;
  }
  return 0;
}
//
/////////////////////////////////////
///////////  Combine Functions //////
/////////////////////////////////////
//
//// Reordering function for float bits
static uint32_t revert_float_bits_dtype32(float number) {
  union {
    float f;
    uint32_t u;
  } value = {.f = number};

  uint32_t sign = (value.u << 8) & 0x80000000;
  uint32_t exponent = (value.u >> 1) & 0x7F800000;
  uint32_t mantissa = (value.u) & 0x7FFFFF;
  return sign | exponent | mantissa;
}
//
//// Helper function to reorder all floats in a bytearray
void revert_all_floats_dtype32(uint8_t *src, size_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  size_t num_floats = len / sizeof(uint32_t);
  for (size_t i = 0; i < num_floats; i++) {
    uint_array[i] = revert_float_bits_dtype32(*(float *)&uint_array[i]);
  }
}

// static int allocate_buffer(uint8_t **result, size_t total_len) {
//   *result = malloc(total_len);
//   if (*result == NULL) {
//     PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
//     return -1;
//   }
//   return 0;
// }
//
// static int handle_combine_mode_220(uint8_t **result, size_t *total_len,
//                                    uint8_t *buf1, uint8_t *buf2, uint8_t
//                                    *buf3, uint8_t *buf4, size_t
//                                    buf1_len, size_t buf2_len, size_t
//                                    buf3_len, size_t buf4_len) {
//   *total_len = buf1_len * 4;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   uint8_t *dst = *result;
//   size_t q_len = *total_len / 4;
//
//   size_t j = 0;
//   for (size_t i = 0; i < q_len; i++) {
//     dst[j++] = buf1[i];
//     dst[j++] = buf2[i];
//     dst[j++] = buf3[i];
//     dst[j++] = buf4[i];
//   }
//   return 0;
// }
//
// static int handle_combine_mode_41(uint8_t **result, size_t *total_len,
//                                   uint8_t *buf1, uint8_t *buf2, uint8_t
//                                   *buf3, uint8_t *buf4, size_t buf1_len,
//                                   size_t buf2_len, size_t buf3_len,
//                                   size_t buf4_len) {
//   *total_len = buf1_len / 3 * 4;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   uint8_t *dst = *result;
//
//   size_t i = 0;
//   for (size_t j = 0; j < *total_len; j += 4) {
//     dst[j] = buf1[i++];
//     dst[j + 1] = buf1[i++];
//     dst[j + 2] = buf1[i++];
//     dst[j + 3] = 0;
//   }
//   return 0;
// }
//
// static int handle_combine_mode_9(uint8_t **result, size_t *total_len,
//                                  uint8_t *buf1, uint8_t *buf2, uint8_t
//                                  *buf3, uint8_t *buf4, size_t buf1_len,
//                                  size_t buf2_len, size_t buf3_len,
//                                  size_t buf4_len) {
//   *total_len = buf1_len * 2;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   uint32_t *dst_uint32 = (uint32_t *)*result;
//   uint16_t *src_uint16 = (uint16_t *)buf1;
//
//   size_t num_uint32 = *total_len / sizeof(uint32_t);
//
//   for (size_t i = 0; i < num_uint32; i++) {
//     dst_uint32[i] = src_uint16[i];
//   }
//
//   return 0;
// }
//
// static int handle_combine_mode_1(uint8_t **result, size_t *total_len,
//                                  uint8_t *buf1, uint8_t *buf2, uint8_t
//                                  *buf3, uint8_t *buf4, size_t buf1_len,
//                                  size_t buf2_len, size_t buf3_len,
//                                  size_t buf4_len) {
//   *total_len = buf1_len * 4;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   uint32_t *dst_uint32 = (uint32_t *)*result;
//   uint8_t *src_uint8 = (uint8_t *)buf1;
//
//   size_t num_uint32 = *total_len / sizeof(uint32_t);
//
//   for (size_t i = 0; i < num_uint32; i++) {
//     dst_uint32[i] = src_uint8[i];
//   }
//
//   return 0;
// }
//
//// Helper function to combine four chunk_buffs into a single bytearray
uint8_t combine_buffers_dtype32(uint8_t *buf1, uint8_t *buf2, uint8_t *buf3,
                                uint8_t *buf4, uint8_t *combinePtr,
                                const size_t *bufLens, int bits_mode,
                                int bytes_mode) {
  uint32_t num_buf = 4;
  uint8_t *bufs[] = {buf1, buf2, buf3, buf4};
  size_t total_len = 0;
  for (uint32_t b = 0; b < num_buf; b++) {
    total_len += bufLens[b];
  }

  size_t q_len = total_len / num_buf;

  uint8_t *dst;
  dst = combinePtr;
  switch (bytes_mode) {
  case 220:
    // 8b1_10_11_100 [decimal 220] - bytegroup to four groups [1,2,3,4]
    for (size_t i = 0; i < q_len; i++) {
      *dst++ = bufs[0][i];
      *dst++ = bufs[1][i];
      *dst++ = bufs[2][i];
      *dst++ = bufs[3][i];
    }
    uint32_t remainder = total_len % num_buf;
    for (uint32_t b = 0; b < num_buf; b++) {
      if (b < remainder) {
        *dst++ = bufs[b][bufLens[b] - 1];
      }
    }

    break;
    //
    //  case 41:
    //    // 8b0_01_01_001 [decimal 41] - truncate the MSB [0,1,1,1]
    //    handle_combine_mode_41(&result, total_len, buf1, buf2, buf3, buf4,
    //    buf1_len,
    //                           buf2_len, buf3_len, buf4_len);
    //    break;
    //
    //  case 9:
    //    //     8b0_00_01_001 [decimal 9] - truncate the MSB+MID_HIGH [0,0,1,1]
    //    handle_combine_mode_9(&result, total_len, buf1, buf2, buf3, buf4,
    //    buf1_len,
    //                          buf2_len, buf3_len, buf4_len);
    //    break;
    //
    //  case 1:
    //    //     8b0_00_00_001 [decimal 1] - truncate the MSB+MID_HIGH+MID_LOW
    //    //     [0,0,0,1]
    //    handle_combine_mode_1(&result, total_len, buf1, buf2, buf3, buf4,
    //    buf1_len,
    //                          buf2_len, buf3_len, buf4_len);
    //    break;
    //
  default:
    PyErr_SetString(
        PyExc_MemoryError,
        "Not supporting bytes_mode for 32bits");
    return -1;
  }
  if (bits_mode == 1) {
    revert_all_floats_dtype32(combinePtr, total_len);
  }
  return 0;
}

/////////////// Helper Function for the Byte ratio /////////////

int buffer_ratio_dtype32(int bytes_mode, uint32_t *buf_ratio) {
  switch (bytes_mode) {
  case 220: // 8b1_10_11_100 - Byte Group to two different groups
    buf_ratio[0] = 4;
    buf_ratio[1] = 4;
    buf_ratio[2] = 4;
    buf_ratio[3] = 4;
    break;

  default:
    // we are not support this splitting bytes_mode
    return -1;
  }
  return 0;
}
