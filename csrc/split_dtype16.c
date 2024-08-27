#define PY_SSIZE_T_CLEAN
#include "huf.h"
#include "split_dtype_functions.h"
#include <Python.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

///////////////////////////////////
/// Split Helper Functions ///////
//////////////////////////////////

// Reordering function for float bits
static uint32_t reorder_float_bits(float number) {
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
static void reorder_all_floats(u_int8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = reorder_float_bits(*(float *)&uint_array[i]);
  }
}

// Helper function to split a bytearray into groups
static int split_bytearray(u_int8_t *src, Py_ssize_t len, u_int8_t **buffers,
                           int bits_mode, int bytes_mode, int is_review,
                           int threads) {
  if (bits_mode == 1) {  // reoreder exponent
    reorder_all_floats(src, len);
  }

  Py_ssize_t half_len = len / 2;
  switch (bytes_mode) {
  case 10:  // 2b01_010 - Byte Group to two different groups
    buffers[0] = PyMem_Malloc(half_len);
    buffers[1] = PyMem_Malloc(half_len);

    if (buffers[0] == NULL || buffers[1] == NULL) {
      PyMem_Free(buffers[0]);
      PyMem_Free(buffers[1]);
      return -1;
    }

    u_int8_t *dst0 = buffers[0];
    u_int8_t *dst1 = buffers[1];

    for (Py_ssize_t i = 0; i < len; i += 2) {
      *dst0++ = src[i];
      *dst1++ = src[i + 1];
    }
    break;

  case 8:  // 4b1000 - Truncate MSByte
           // We are refering to the MSBbyte as little endian, thus we omit buf2
  case 1:  // 4b1000 - Truncate LSByte
    // We are refering to the LSByte  as a little endian, thus we omit buf1
    buffers[0] = PyMem_Malloc(half_len);
    buffers[1] = NULL;

    if (buffers[0] == NULL) {
      PyMem_Free(buffers[0]);
      return -1;
    }

    dst0 = buffers[0];

    if (bytes_mode == 1) {
      for (Py_ssize_t i = 0; i < len; i += 2) {
        *dst0++ = src[i];
      }
    } else {
      for (Py_ssize_t i = 0; i < len; i += 2) {
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
static uint32_t revert_float_bits(float number) {
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
static void revert_all_floats(u_int8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = revert_float_bits(*(float *)&uint_array[i]);
  }
}

// Helper function to combine four buffers into a single bytearray
static int combine_buffers(u_int8_t *buf1, u_int8_t *buf2, u_int8_t *combinePtr,
                           Py_ssize_t half_len, int bits_mode, int bytes_mode,
                           int threads) {
  Py_ssize_t total_len = half_len * 2;

  u_int8_t *dst;
  dst = combinePtr;

  switch (bytes_mode) {
  case 10: // 2b01_010 - Byte Group to two different groups
    for (Py_ssize_t i = 0; i < half_len; i++) {
      *dst++ = buf1[i];
      *dst++ = buf2[i];
    }
    break;

  case 8: // 4b1000 - Truncate MSByte
          // We are refering to the MSByte as a little endian, thus we omit buf2
  case 1: // 4b001 - Truncate LSByte
          // We are refering to the LSByte as a little endian, thus we omit buf1

    if (bytes_mode == 8) {
      for (Py_ssize_t i = 0; i < half_len; i++) {
        *dst++ = 0;
        *dst++ = buf1[i];
      }
    } else {
      for (Py_ssize_t i = 0; i < half_len; i++) {
        *dst++ = buf1[i];
        *dst++ = 0;
      }
    }
    break;

  default:
    // we are not supporting this splitting bytes_mode
    return -1;
  }
  // printf("combinePtr %zu\n ",combinePtr);
  // printf("dst %zu\n ", dst);
  //  Revert the reordering of all floats if needed
  if (bits_mode == 1) {
    revert_all_floats(combinePtr, total_len);
  }
  return 0;
}

///////////// helper function to prepare the split data
//////////////////////////////////////
u_int8_t *prepare_split_results(size_t header_len, size_t numBuf,
                                size_t numChunks, u_int8_t *header,
                                u_int8_t *compressedData[numBuf][numChunks],
                                uint32_t compChunksSize[numBuf][numChunks],
                                u_int8_t compChunksType[numBuf][numChunks],
                                size_t cumulativeChunksSize[numBuf][numChunks],
                                size_t *totalCompressedSize,
                                size_t *resBufSize) {
  *resBufSize = header_len;
  size_t compChunksTypeLen =
      numBuf * numChunks * (sizeof(compChunksType[numBuf][numChunks]));
  size_t cumulativeChunksSizeLen =
      numBuf * numChunks * (sizeof(cumulativeChunksSize[numBuf][numChunks]));
  *resBufSize += compChunksTypeLen;
  *resBufSize += cumulativeChunksSizeLen;
  for (size_t b = 0; b < numBuf; b++) {
    *resBufSize += totalCompressedSize[b];
  }

  // update compress_buffer_len
  memcpy(&header[24], resBufSize, sizeof(size_t));

  u_int8_t *resultBuf = PyMem_Malloc(*resBufSize);
  if (!resultBuf) {
    PyErr_SetString(
        PyExc_MemoryError,
        "Failed to allocate memory for result buffer in split function");
    PyMem_Free(resultBuf);
    return NULL;
  }

  // Copy data to result buffer
  size_t offset = 0;
  memcpy(resultBuf + offset, header, header_len);
  offset += header_len;
  memcpy(resultBuf + offset, compChunksType, compChunksTypeLen);
  offset += compChunksTypeLen;
  memcpy(resultBuf + offset, cumulativeChunksSize, cumulativeChunksSizeLen);
  offset += cumulativeChunksSizeLen;

  for (uint32_t b = 0; b < numBuf; b++) {
    for (uint32_t c = 0; c < numChunks; c++) {
      memcpy(resultBuf + offset, compressedData[b][c], compChunksSize[b][c]);
      // printf("\n\n resultBuf + offset %zu\n\n\n ", resultBuf + offset);
      offset += compChunksSize[b][c];
    }
  }

  return resultBuf;
}

/////////////////////////////////////////////////////////////
//////////////// Python callable Functions /////////////////
/////////////////////////////////////////////////////////////

// Python callable function to split a bytearray into four buffers
// bits_mode:
//     0 - no ordering of the bits
//     1 - reorder of the exponent (eponent, sign_bit, mantissa)
// bytes_mode:
//     [we are refering to the bytes order as first 2bits refer to the MSByte
//     and the second two bits to the LSByte] 2b [MSB Byte],2b[LSB Byte] 0 -
//     truncate this byte 1 or 2 - a group of bytes 4b0110 [6] - bytegroup to
//     two groups 4b0001 [1] - truncate the MSByte 4b1000 [8] - truncate the
//     LSByte
// is_review:
//     Even if you have the Byte mode, you can change it if needed.
//     0 - No review, take the bit_mode and byte_mode
//     1 - the finction can change the Bytes_mode

PyObject *py_split_dtype16(PyObject *self, PyObject *args) {
  const uint32_t numBuf = 2;
  Py_buffer header, data;
  int bits_mode, bytes_mode, is_redata, checkThAfterPercent, threads;
  size_t bgChunkSize;
  float compThreshold;
  // u_int8_t isPrint = 0;
  // clock_t startTime, endTime, startBGTime, endBGTime;
  // startTime = clock();

  if (!PyArg_ParseTuple(args, "y*y*iiinfii", &header, &data, &bits_mode,
                        &bytes_mode, &is_redata, &bgChunkSize, &compThreshold,
                        &checkThAfterPercent, &threads)) {
    return NULL;
  }

  // Byte Group per chunk, Compress per bufChunk
  size_t numChunks = (data.len + bgChunkSize - 1) / bgChunkSize;
  u_int8_t *buffers[numChunks][numBuf];
  size_t curChunk = 0;

  u_int8_t *compressedData[numBuf][numChunks];
  uint32_t compChunksSize[numBuf][numChunks];
  uint32_t unCompChunksSize[numChunks];
  size_t totalCompressedSize[] = {0, 0};
  size_t totalUnCompressedSize[] = {0, 0};
  u_int8_t compChunksType[numBuf][numChunks];
  size_t cumulativeChunksSize[numBuf][numChunks];
  u_int8_t isThCheck[] = {0, 0};
  u_int8_t noNeedToCompress[] = {0, 0};
  uint32_t checkCompTh =
      (uint32_t)ceil((double)numChunks / checkThAfterPercent);
  // if (isPrint) {
  //     startBGTime = clock();
  //  }

  /////////// start multi Threading - Each chunk to different thread
  //////////////////
  for (size_t offset = 0; offset < (size_t)data.len; offset += bgChunkSize) {
    size_t curBgChunkSize =
        (data.len - offset > bgChunkSize) ? bgChunkSize : (data.len - offset);

    //    printf ("offset %zu\n", offset);
    size_t curCompChunkSize = curBgChunkSize / numBuf;
    unCompChunksSize[curChunk] = curCompChunkSize;
    // Byte Grouping + Byte Ordering

    if (split_bytearray(data.buf + offset, curBgChunkSize, buffers[curChunk],
                        bits_mode, bytes_mode, is_redata, threads) != 0) {
      PyBuffer_Release(&data);
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
      return NULL;
    }
    //    if (isPrint) {
    //      endBGTime = clock();
    //      bgTime = (double)(endBGTime - startBGTime) / CLOCKS_PER_SEC;
    //    }
    // Compression on each Buf

    for (uint32_t b = 0; b < numBuf; b++) {
      if (isThCheck[b] == 0 &&
          curChunk >=
              checkCompTh) {  // check that we really need to compress this buf
        if (totalCompressedSize[b] * 1.0 >
            totalUnCompressedSize[b] * compThreshold) {
          isThCheck[b] = 1;
          noNeedToCompress[b] = 1;
        }
      }

      compressedData[b][curChunk] = PyMem_Malloc(bgChunkSize);
      if (!compressedData[b][curChunk]) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        for (uint32_t j = 0; j < numBuf; j++) {
          for (uint32_t c = 0; c < curChunk - 1; c++) {
            PyMem_Free(compressedData[j][c]);
          }
          for (uint32_t j = 0; j < b; j++) {
            PyMem_Free(compressedData[j][curChunk]);
          }
        }
        return NULL;
      }

      if (buffers[curChunk][b] != NULL) {
        if (noNeedToCompress[b] == 0) {
          compChunksSize[b][curChunk] =
              HUF_compress(compressedData[b][curChunk], bgChunkSize,
                           buffers[curChunk][b], curCompChunkSize);
        } else {
          compChunksSize[b][curChunk] = 0;
        }

        if (compChunksSize[b][curChunk] != 0 &&
            (compChunksSize[b][curChunk] <
             unCompChunksSize[curChunk] * compThreshold)) {
          compChunksType[b][curChunk] = 1;  // Compress with Huffman
        } else {                            // the buffer was not compressed
          PyMem_Free(compressedData[b][curChunk]);
          compChunksSize[b][curChunk] = unCompChunksSize[curChunk];
          compChunksType[b][curChunk] = 0;  // not compressed
          compressedData[b][curChunk] = buffers[curChunk][b];
        }
        totalCompressedSize[b] += compChunksSize[b][curChunk];
        totalUnCompressedSize[b] += unCompChunksSize[curChunk];
        cumulativeChunksSize[b][curChunk] = totalCompressedSize[b];
      }

    }  // end for loop -> compression
    curChunk++;
  }  // end for loop - chunk
  ////////////// The end of multi Threading part 1
  /////////////////////////////////

  // endTime = clock();
  // double compressTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  PyObject *result;
  u_int8_t *resultBuf;
  size_t resBufSize;

  resultBuf = prepare_split_results(
      header.len, numBuf, numChunks, header.buf, compressedData, compChunksSize,
      compChunksType, cumulativeChunksSize, totalCompressedSize, &resBufSize);
  if (resultBuf == NULL) {
    // Free all Mallocs
    // print Error
    return NULL;
  }

  result = Py_BuildValue("y#", resultBuf, resBufSize);

  // Freeing compressedData array
  for (uint32_t c = 0; c < numChunks; c++) {
    for (uint32_t b = 0; b < numBuf; b++) {
      if (buffers[c][b] != NULL) {
        PyMem_Free(buffers[c][b]);
      }
    }
  }

  for (uint32_t c = 0; c < numChunks; c++) {
    for (uint32_t b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] == 1) {
        PyMem_Free(compressedData[b][c]);
      }
    }
  }
  PyBuffer_Release(&header);
  PyBuffer_Release(&data);
  PyMem_Free(resultBuf);
  return result;
}

// Python callable function to combine four buffers into a single bytearray
PyObject *py_combine_dtype16(PyObject *self, PyObject *args) {
  Py_buffer data;

  int bits_mode, bytes_mode, threads;
  uint32_t numBuf = 2;
  size_t bgChunkSize, origSize;

  if (!PyArg_ParseTuple(args, "y*iinni", &data, &bits_mode, &bytes_mode,
                        &bgChunkSize, &origSize, &threads)) {
    return NULL;
  }

  size_t numChunks = (origSize + bgChunkSize - 1) / bgChunkSize;
  uint32_t compChunkSize = bgChunkSize / numBuf;

  u_int8_t *ptrChunksType = (u_int8_t *)data.buf;
  size_t *ptrChunksCumulative = (size_t *)(ptrChunksType + numBuf * numChunks);
  u_int8_t *ptrCompressData[numBuf];
  ptrCompressData[0] = (u_int8_t *)(ptrChunksCumulative + numBuf * numChunks);
  size_t cumulativeChunksSize[numBuf][numChunks];
  uint32_t compChunksType[numBuf][numChunks];
  size_t compCumulativeChunksPos[numBuf][numChunks + 1];
  size_t CompChunksLen[numBuf][numChunks];
  u_int8_t *resultBuf = NULL;
  u_int8_t *deCompressedData[numBuf][numChunks];
  size_t decompLen[numChunks];
  // clock_t startTime, endTime;
  // startTime = clock();

  // Preparation for decompression
  compCumulativeChunksPos[0][0] = 0;
  compCumulativeChunksPos[1][0] = 0;

  for (uint32_t b = 0; b < numBuf; b++) {
    for (uint32_t c = 0; c < numChunks; c++) {
      compChunksType[b][c] = (*ptrChunksType++);
      cumulativeChunksSize[b][c] = (*ptrChunksCumulative++);
    }
  }

  for (uint32_t c = 0; c < numChunks; c++) {
    for (uint32_t b = 0; b < numBuf; b++) {
      compCumulativeChunksPos[b][c + 1] = cumulativeChunksSize[b][c];
      CompChunksLen[b][c] =
          compCumulativeChunksPos[b][c + 1] - compCumulativeChunksPos[b][c];
    }
  }

  for (size_t c = 0; c < numChunks; c++) {
    for (uint32_t b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] == 0) { // no compression is needed
      } else {
        if (compChunksType[b][c] == 1) { // open with Huffman compression
        } else {
          PyErr_SetString(
              PyExc_MemoryError,
              "Compress Type is not correct in Decompression function");
          return NULL;
        }
      }
    }
  }

  ptrCompressData[1] =
      ptrCompressData[0] + cumulativeChunksSize[0][numChunks - 1];
  for (uint32_t c = 0; c < numChunks; c++) {
    if (c < numChunks - 1) {
      decompLen[c] = compChunkSize;
    } else {
      decompLen[c] =
          (size_t)(origSize / numBuf - compChunkSize * (numChunks - 1));
    }
  }
  resultBuf = PyMem_Malloc(origSize);
  if (!resultBuf) {
    PyErr_SetString(
        PyExc_MemoryError,
        "Failed to allocate memory for result buffer in split function");
    PyMem_Free(resultBuf);
    return NULL;
  }

  // endTime = clock();
  // double decompressTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  ////////////// Multi threading /////////////////////////////

  for (size_t c = 0; c < numChunks; c++) {
    // decompress
    for (uint32_t b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] == 0) {  // No Need to compression
        deCompressedData[b][c] =
            ptrCompressData[b] + compCumulativeChunksPos[b][c];
      } else if (compChunksType[b][c] == 1) {  // decompress using Huffman
        deCompressedData[b][c] = PyMem_Malloc(decompLen[c]);
        if (deCompressedData[b][c] == NULL) {
          PyErr_SetString(
              PyExc_MemoryError,
              "Failed to allocate memory - Function during decompression");
          free(deCompressedData[b][c]);
          return NULL;
        }
        size_t decompressedSize = HUF_decompress(
            deCompressedData[b][c], decompLen[c],
            (void *)(ptrCompressData[b] + compCumulativeChunksPos[b][c]),
            CompChunksLen[b][c]);

        if (HUF_isError(decompressedSize)) {
          HUF_getErrorName(decompressedSize);
          return 0;
        }

        if (decompressedSize != decompLen[c]) {
          return 0;
        }
      }
    }

    // Combine
    u_int8_t *combinePtr = resultBuf + bgChunkSize * c;
    if (combine_buffers(deCompressedData[0][c], deCompressedData[1][c],
                        combinePtr, decompLen[c], bits_mode, bytes_mode,
                        threads) != 0) {
      return NULL;
    }
  }
  ////////////// Finish Multi threading /////////////////////////////

  PyObject *py_result =
      PyByteArray_FromStringAndSize((const char *)resultBuf, origSize);

  for (size_t c = 0; c < numChunks; c++) {
    for (uint32_t b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] > 0) {
        free(deCompressedData[b][c]);
      }
    }
  }

  PyMem_Free(resultBuf);
  PyBuffer_Release(&data);
  return py_result;
}
