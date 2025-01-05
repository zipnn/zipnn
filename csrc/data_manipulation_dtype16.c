#include <Python.h>
#include <stdint.h>
#include <time.h>

///////////////////////////////////
/// Split Helper Functions ///////
//////////////////////////////////

// Reordering function for float bits
static uint32_t reorder_float_bits_dtype16(float number) {
  union {
    float f;
    uint32_t u;
  } value = {.f = number};

  uint32_t sign = (value.u >> 8) & 0x800080;
  uint32_t exponent = (value.u << 1) & 0xFF00FF00;
  uint32_t mantissa = (value.u) & 0x7F007F;
  return exponent | sign | mantissa;
}

// Helper function to reorder all floats in a bytearray
static void reorder_all_floats_dtype16(uint8_t *src, size_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  size_t num_floats = len / sizeof(uint32_t);
  for (size_t i = 0; i < num_floats; i++) {
    uint_array[i] = reorder_float_bits_dtype16(*(float *)&uint_array[i]);
  }
}

// Helper function to split a bytearray into groups
int split_bytearray_dtype16(uint8_t *src, size_t len, uint8_t **chunk_buffs,
                            size_t *unCompChunksSizeCurChunk, int bits_mode,
                            int bytes_mode, int is_review) {
  if (bits_mode == 1) { // reoreder exponent
    reorder_all_floats_dtype16(src, len);
  }
  size_t half_len = len / 2;
  size_t lens[] = {half_len, half_len};
  int remainder = len % 2;
  if (remainder > 0) {
    lens[0] += 1;
  }

  switch (bytes_mode) {
  case 10: // 2b01_010 - Byte Group to two different groups
    chunk_buffs[0] = malloc(lens[0]);
    chunk_buffs[1] = malloc(lens[1]);
    unCompChunksSizeCurChunk[0] = lens[0];
    unCompChunksSizeCurChunk[1] = lens[1];

    if (chunk_buffs[0] == NULL || chunk_buffs[1] == NULL) {
      free(chunk_buffs[0]);
     PyErr_SetString(PyExc_MemoryError,
                    "Failed to allocate memory, allocate 2 buffers");
     free(chunk_buffs[1]);
      return -1;
    }

    uint8_t *dst0 = chunk_buffs[0];
    uint8_t *dst1 = chunk_buffs[1];

    for (size_t i = 0; i < len; i += 2) {
      *dst0++ = src[i];
      *dst1++ = src[i + 1];
    }
    if (remainder > 0) {
      *dst0 = src[len - 1];
    }
    break;

  case 8: // 4b1000 - Truncate MSByte
          // We are refering to the MSBbyte as little endian, thus we omit buf2
  case 1: // 4b1000 - Truncate LSByte
    // We are refering to the LSByte  as a little endian, thus we omit buf1
    chunk_buffs[0] = malloc(half_len);
    chunk_buffs[1] = NULL;
    unCompChunksSizeCurChunk[0] = half_len;
    unCompChunksSizeCurChunk[1] = 0;

    if (chunk_buffs[0] == NULL) {
       PyErr_SetString(PyExc_MemoryError,
            "Failed to allocate memory, allocate 1 buffer");
       free(chunk_buffs[0]);
      return -1;
    }

    dst0 = chunk_buffs[0];

    if (bytes_mode == 1) {
      for (size_t i = 0; i < len; i += 2) {
        *dst0++ = src[i];
      }
    } else {
      for (size_t i = 0; i < len; i += 2) {
        *dst0++ = src[i + 1];
      }
    }
    break;

  default:
    // we are not support this splitting bytes_mode
    return -1;
  }
  return 0;
}

///////////////////////////////////
/////////  Combine Functions //////
///////////////////////////////////

// Reordering function for float bits
static uint32_t revert_float_bits_dtype16(float number) {
  union {
    float f;
    uint32_t u;
  } value = {.f = number};

  uint32_t sign = (value.u << 8) & 0x80008000;
  uint32_t exponent = (value.u >> 1) & 0x7F807F80;
  uint32_t mantissa = (value.u) & 0x7F007F;
  return sign | exponent | mantissa;
}

// Helper function to reorder all floats in a bytearray
static void revert_all_floats_dtype16(uint8_t *src, size_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  size_t num_floats = len / sizeof(uint32_t);
  for (size_t i = 0; i < num_floats; i++) {
    uint_array[i] = revert_float_bits_dtype16(*(float *)&uint_array[i]);
  }
}

// Helper function to combine four chunk_buffs into a single bytearray
int combine_buffers_dtype16(const uint8_t *buf1, const uint8_t *buf2,
                            uint8_t *combinePtr, const size_t *bufLens,
                            int bits_mode, int bytes_mode) {
  size_t total_len = bufLens[0] + bufLens[1];
  size_t half_len = total_len / 2;

  uint8_t *dst;
  dst = combinePtr;

  switch (bytes_mode) {
  case 10: // 2b01_010 - Byte Group to two different groups
    for (size_t i = 0; i < half_len; i++) {
      *dst++ = buf1[i];
      *dst++ = buf2[i];
    }
    if (bufLens[0] > bufLens[1]) { // There is a remainder
      *dst = buf1[bufLens[0] - 1];
    }
    break;

  case 8: // 4b1000 - Truncate MSByte
          // We are refering to the MSByte as a little endian, thus we omit buf2
  case 1: // 4b001 - Truncate LSByte
          // We are refering to the LSByte as a little endian, thus we omit buf1

    if (bytes_mode == 8) {
      for (size_t i = 0; i < bufLens[0]; i++) {
        *dst++ = 0;
        *dst++ = buf1[i];
      }
    } else {
      for (size_t i = 0; i < bufLens[0]; i++) {
        *dst++ = buf1[i];
        *dst++ = 0;
      }
    }
    break;

  default:
    PyErr_SetString(
        PyExc_MemoryError,
        "Not supporting bytes_mode for 16bits");
    return -1;
  }
  //  Revert the reordering of all floats if needed
  if (bits_mode == 1) {
    revert_all_floats_dtype16(combinePtr, total_len);
  }
  return 0;
}

/////////////// Helper Function for the Byte ratio /////////////

int buffer_ratio_dtype16(int bytes_mode, uint32_t *buf_ratio) {
  switch (bytes_mode) {
  case 10: // 2b01_010 - Byte Group to two different groups
    buf_ratio[0] = 2;
    buf_ratio[1] = 2;
    break;

  case 8: // 4b1000 - Truncate MSByte
  case 1: // 4b1000 - Truncate LSByte
    buf_ratio[0] = 1;
    buf_ratio[1] = 0;
    break;

  default:
    // we are not support this splitting bytes_mode
    return -1;
  }
  return 0;
}
