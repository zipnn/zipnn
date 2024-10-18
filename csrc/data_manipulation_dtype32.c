#include <Python.h>
#include <stdint.h>
#include <time.h>
#include "methods_enums.h"
#include "methods_utils.h"

//// Helper function that count zero bytes

static int update_seq_count(size_t *lastZero, size_t *tmpSeqZeros, size_t *maxSeqZeros, int num_buf) {
  for (int b=0; b < num_buf; b++) {
    if (lastZero[b]) {
      (tmpSeqZeros[b])++;
      if (tmpSeqZeros > maxSeqZeros) {
        maxSeqZeros[b] = tmpSeqZeros[b];	      
      }      
    }
    else {
      tmpSeqZeros[b] = 0;	    
    }	    
  }
}

static int count_zero_bytes_dtype32(const u_int8_t *src, Py_ssize_t len,
                             size_t* zeroCount, size_t* maxSeqZeros, const int num_buf) {
  int is_print_zeros = 0;
  Py_ssize_t num_uint32 =
      len /
      sizeof(
          uint32_t);  // Calculate how many uint32_t elements are in the buffer
  const uint32_t *uint32_array =
      (uint32_t *)src;  // Cast the byte buffer to a uint32_t array
  
  size_t tmpSeqZeros[] = {0, 0, 0, 0}; 
  for (int b=0; b < num_buf; b++) {
    zeroCount[b] = 0;
    maxSeqZeros[b] = 0;    
  }
   
  for (size_t i = 0; i < num_uint32; i++) {
    uint32_t value = uint32_array[i];
    size_t lastZero[] = {0, 0, 0, 0};
    if ((value & 0xFF000000) == 0) {
      (zeroCount[3])++;
      lastZero[3] = 1;
    }
    if ((value & 0xFF0000) == 0) {
      (zeroCount[2])++;
      lastZero[2] = 1;
    }
    if ((value & 0xFF00) == 0) {
      (zeroCount[1])++;
      lastZero[1] = 1;
    }
    if (value == 0) {
      (zeroCount[0])++;
      lastZero[0] = 1;
    }
    update_seq_count(lastZero, tmpSeqZeros, maxSeqZeros, num_buf);
  }
  if(is_print_zeros) { 
    for (int b=0; b < num_buf; b++) {
      float zeroCountPrecent =  zeroCount[b] *1.0 / (num_uint32); 
      float zeroSeqPrecent =  maxSeqZeros[b] *1.0 / (num_uint32); 
      printf ("Group[%d] zeroCount %zu [%.4f], maxSeqZeros %zu [%.4f] \n", b, zeroCount[b], zeroCountPrecent, maxSeqZeros[b], zeroSeqPrecent);
    }
  }
  return 0;
}

 
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
void reorder_all_floats_dtype32(u_int8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
//  printf ("num_floats %zu", num_floats);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = reorder_float_bits_dtype32(*(float *)&uint_array[i]);
  }
}

//
int allocate_4chunk_buffs(u_int8_t **chunk_buffs, Py_ssize_t *bufLens,
                          uint32_t num_buf) {
  chunk_buffs[0] = (bufLens[0] > 0) ? PyMem_Malloc(bufLens[0]) : NULL;
  chunk_buffs[1] = (bufLens[1] > 0) ? PyMem_Malloc(bufLens[1]) : NULL;
  chunk_buffs[2] = (bufLens[2] > 0) ? PyMem_Malloc(bufLens[2]) : NULL;
  chunk_buffs[3] = (bufLens[3] > 0) ? PyMem_Malloc(bufLens[3]) : NULL;

//  for (int i=0; i<4;i++){
//    printf("bufLens[i] %zu\n", bufLens[i]);
//  } 

  if ((bufLens[0] > 0 && chunk_buffs[0] == NULL) ||
      (bufLens[1] > 0 && chunk_buffs[1] == NULL) ||
      (bufLens[2] > 0 && chunk_buffs[2] == NULL) ||
      (bufLens[3] > 0 && chunk_buffs[3] == NULL)) {
    PyMem_Free(chunk_buffs[0]);
    PyMem_Free(chunk_buffs[1]);
    PyMem_Free(chunk_buffs[2]);
    PyMem_Free(chunk_buffs[3]);
    PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory, allocate_4chunk_buffs");
    return -1;
  }
  return 0;
}

int handle_split_mode_220(const u_int8_t *src, Py_ssize_t total_len,
                          u_int8_t **chunk_buffs, Py_ssize_t *bufLens,
                          uint32_t num_buf) {
  Py_ssize_t q_len = total_len / num_buf;
  int remainder = total_len % num_buf;

  for (Py_ssize_t b = 0; b < num_buf; b++) {
    if (b < remainder) {
      bufLens[b] = q_len + 1;
    } else {
      bufLens[b] = q_len;
    }
  }

  if (allocate_4chunk_buffs(chunk_buffs, bufLens, num_buf) != 0)
    return -1;

  u_int8_t *dst1 = chunk_buffs[0], *dst2 = chunk_buffs[1],
           *dst3 = chunk_buffs[2], *dst4 = chunk_buffs[3];

  for (Py_ssize_t i = 0; i < total_len; i += 4) {
    *dst1++ = src[i];
    *dst2++ = src[i + 1];
    *dst3++ = src[i + 2];
    *dst4++ = src[i + 3];
  }
  
//  printf ("src[0] %zu\n", src[0]);
//  printf ("src[1] %zu\n", src[1]);
//  printf ("src[2] %zu\n", src[2]);
//  printf ("chunk_buffs[0][0] %zu\n", chunk_buffs[0][0]);
//  printf ("chunk_buffs[1][0] %zu\n", chunk_buffs[1][0]);
//  printf ("chunk_buffs[2][0] %zu\n", chunk_buffs[2][0]);
  if (remainder > 3) {
     size_t total_len_remainder	= total_len - remainder;
     chunk_buffs[0][q_len] = src[total_len_remainder];
     if (remainder > 1) {
      chunk_buffs[1][q_len] = src[total_len_remainder + 1];
     }
     if (remainder > 2) {
      chunk_buffs[2][q_len] = src[total_len_remainder + 2];
     }
  }
//  printf ("chunk_buffs[0][0] %zu\n", chunk_buffs[0][0]);
//  printf ("chunk_buffs[1][0] %zu\n", chunk_buffs[1][0]);
//  printf ("chunk_buffs[2][0] %zu\n", chunk_buffs[2][0]);
  return 0;
}
//
// int handle_split_mode_41(u_int8_t *src, Py_ssize_t total_len,
//                                u_int8_t **buf1, u_int8_t **buf2, u_int8_t
//                                **buf3, u_int8_t **buf4, Py_ssize_t *buf1_len,
//                                Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
//                                Py_ssize_t *buf4_len) {
//  Py_ssize_t three_q_len = total_len / 4 * 3;
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
//  u_int8_t *dst1 = *buf1;
//
//  Py_ssize_t j = 0;
//  for (Py_ssize_t i = 0; i < total_len; i += 4) {
//    dst1[j++] = src[i];
//    dst1[j++] = src[i + 1];
//    dst1[j++] = src[i + 2];
//  }
//  return 0;
//}
//
// int handle_split_mode_9(u_int8_t *src, Py_ssize_t total_len,
//                               u_int8_t **buf1, u_int8_t **buf2, u_int8_t
//                               **buf3, u_int8_t **buf4, Py_ssize_t *buf1_len,
//                               Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
//                               Py_ssize_t *buf4_len) {
//  Py_ssize_t half_len = total_len / 2;
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
//  Py_ssize_t num_uint32 = total_len / sizeof(uint32_t);
//
//  for (Py_ssize_t i = 0; i < num_uint32; i++) {
//    dst_uint16[i] = (uint16_t)src_uint32[i];
//  }
//
//  return 0;
//}
//
// static int handle_split_mode_1(u_int8_t *src, Py_ssize_t total_len,
//                               u_int8_t **buf1, u_int8_t **buf2, u_int8_t
//                               **buf3, u_int8_t **buf4, Py_ssize_t *buf1_len,
//                               Py_ssize_t *buf2_len, Py_ssize_t *buf3_len,
//                               Py_ssize_t *buf4_len) {
//  Py_ssize_t half_len = total_len / 4;
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
//  u_int8_t *dst_uint8 = (u_int8_t *)*buf1;
//
//  Py_ssize_t num_uint32 = total_len / sizeof(uint32_t);
//
//  for (Py_ssize_t i = 0; i < num_uint32; i++) {
//    dst_uint8[i] = (u_int8_t)src_uint32[i];
//  }
//
//  return 0;
//}
//
//// Helper function to split a bytearray into four chunk_buffs
int split_bytearray_dtype32(u_int8_t *src, Py_ssize_t len,
                            u_int8_t **chunk_buffs, size_t *bufLens,
                            int bits_mode, int bytes_mode, int method, int *chunk_methods, int is_review,
                            int threads) {
  const int num_buf = 4;
  for (int i = 0; i < num_buf; i++) {
    chunk_methods[i] = method;
  }

  if (bits_mode == 1) {  // reoreder exponent
    reorder_all_floats_dtype32(src, len);
  }

  int is_print_zero_time = 0;

  if (method == AUTO) {
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    size_t zeroCount[num_buf];
    size_t maxSeqZeros[num_buf];
    count_zero_bytes_dtype32(src, len, zeroCount, maxSeqZeros, num_buf);
    calc_chunk_methods_dtype32(zeroCount, maxSeqZeros, chunk_methods, (size_t)(len/num_buf), num_buf);
    end = clock();  // End the timer
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    if (is_print_zero_time) {
       printf ("count_zero_bytes_dtype32 %f\n", cpu_time_used);
    }
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
void revert_all_floats_dtype32(u_int8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = revert_float_bits_dtype32(*(float *)&uint_array[i]);
  }
}

// static int allocate_buffer(u_int8_t **result, Py_ssize_t total_len) {
//   *result = PyMem_Malloc(total_len);
//   if (*result == NULL) {
//     PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
//     return -1;
//   }
//   return 0;
// }
//
// static int handle_combine_mode_220(u_int8_t **result, Py_ssize_t *total_len,
//                                    u_int8_t *buf1, u_int8_t *buf2, u_int8_t
//                                    *buf3, u_int8_t *buf4, Py_ssize_t
//                                    buf1_len, Py_ssize_t buf2_len, Py_ssize_t
//                                    buf3_len, Py_ssize_t buf4_len) {
//   *total_len = buf1_len * 4;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   u_int8_t *dst = *result;
//   Py_ssize_t q_len = *total_len / 4;
//
//   Py_ssize_t j = 0;
//   for (Py_ssize_t i = 0; i < q_len; i++) {
//     dst[j++] = buf1[i];
//     dst[j++] = buf2[i];
//     dst[j++] = buf3[i];
//     dst[j++] = buf4[i];
//   }
//   return 0;
// }
//
// static int handle_combine_mode_41(u_int8_t **result, Py_ssize_t *total_len,
//                                   u_int8_t *buf1, u_int8_t *buf2, u_int8_t
//                                   *buf3, u_int8_t *buf4, Py_ssize_t buf1_len,
//                                   Py_ssize_t buf2_len, Py_ssize_t buf3_len,
//                                   Py_ssize_t buf4_len) {
//   *total_len = buf1_len / 3 * 4;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   u_int8_t *dst = *result;
//
//   Py_ssize_t i = 0;
//   for (Py_ssize_t j = 0; j < *total_len; j += 4) {
//     dst[j] = buf1[i++];
//     dst[j + 1] = buf1[i++];
//     dst[j + 2] = buf1[i++];
//     dst[j + 3] = 0;
//   }
//   return 0;
// }
//
// static int handle_combine_mode_9(u_int8_t **result, Py_ssize_t *total_len,
//                                  u_int8_t *buf1, u_int8_t *buf2, u_int8_t
//                                  *buf3, u_int8_t *buf4, Py_ssize_t buf1_len,
//                                  Py_ssize_t buf2_len, Py_ssize_t buf3_len,
//                                  Py_ssize_t buf4_len) {
//   *total_len = buf1_len * 2;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   uint32_t *dst_uint32 = (uint32_t *)*result;
//   uint16_t *src_uint16 = (uint16_t *)buf1;
//
//   Py_ssize_t num_uint32 = *total_len / sizeof(uint32_t);
//
//   for (Py_ssize_t i = 0; i < num_uint32; i++) {
//     dst_uint32[i] = src_uint16[i];
//   }
//
//   return 0;
// }
//
// static int handle_combine_mode_1(u_int8_t **result, Py_ssize_t *total_len,
//                                  u_int8_t *buf1, u_int8_t *buf2, u_int8_t
//                                  *buf3, u_int8_t *buf4, Py_ssize_t buf1_len,
//                                  Py_ssize_t buf2_len, Py_ssize_t buf3_len,
//                                  Py_ssize_t buf4_len) {
//   *total_len = buf1_len * 4;
//   if (allocate_buffer(result, *total_len) != 0)
//     return -1;
//
//   uint32_t *dst_uint32 = (uint32_t *)*result;
//   u_int8_t *src_uint8 = (u_int8_t *)buf1;
//
//   Py_ssize_t num_uint32 = *total_len / sizeof(uint32_t);
//
//   for (Py_ssize_t i = 0; i < num_uint32; i++) {
//     dst_uint32[i] = src_uint8[i];
//   }
//
//   return 0;
// }
//
//// Helper function to combine four chunk_buffs into a single bytearray
u_int8_t combine_buffers_dtype32(u_int8_t *buf1, u_int8_t *buf2, u_int8_t *buf3,
                                 u_int8_t *buf4, u_int8_t *combinePtr,
                                 const Py_ssize_t *bufLens, int bits_mode,
                                 int bytes_mode, int threads) {
  int num_buf = 4;
  u_int8_t *bufs[] = {buf1, buf2, buf3, buf4};
  size_t total_len = 0;
  for (int b = 0; b < num_buf; b++) {
    total_len += bufLens[b];
  }

  size_t q_len = total_len / num_buf;

  u_int8_t *dst;
  dst = combinePtr;
  switch (bytes_mode) {
  case 220:
    // 8b1_10_11_100 [decimal 220] - bytegroup to four groups [1,2,3,4]
    for (Py_ssize_t i = 0; i < q_len; i++) {
      *dst++ = bufs[0][i];
      *dst++ = bufs[1][i];
      *dst++ = bufs[2][i];
      *dst++ = bufs[3][i];
    }
    int remainder = total_len % num_buf;
    for (Py_ssize_t b = 0; b < num_buf; b++) {
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
    // we are not supportin this splitting bytes_mode
    return 0;
  }
  if (bits_mode == 1) {
    revert_all_floats_dtype32(combinePtr, total_len);
  }
  return 0;
}

/////////////// Helper Function for the Byte ratio /////////////

int buffer_ratio_dtype32(int bytes_mode, uint32_t *buf_ratio) {
  switch (bytes_mode) {
  case 220:  // 8b1_10_11_100 - Byte Group to two different groups
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
