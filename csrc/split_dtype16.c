#define PY_SSIZE_T_CLEAN
#include "split_dtype_functions.h"
#include <Python.h>
#include <stdint.h>
#include <time.h>
#include "huf_cover.h"
#include "huf.h"

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
static void reorder_all_floats(uint8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = reorder_float_bits(*(float *)&uint_array[i]);
  }
}

// Helper function to split a bytearray into four buffers
static int split_bytearray(uint8_t *src, Py_ssize_t len, uint8_t **buffers,
                           int bits_mode, int bytes_mode, int is_review,
                           int threads) {

  if (bits_mode == 1) { // reoreder exponent
    reorder_all_floats(src, len);
  }

  Py_ssize_t half_len = len / 2;
  switch (bytes_mode) {
  case 6: // 2b0110 - Byte Group to two different groups
    buffers[0] = PyMem_Malloc(half_len);
    buffers[1] = PyMem_Malloc(half_len);

    if (buffers[0] == NULL || buffers[1] == NULL) {
      PyMem_Free(buffers[0]);
      PyMem_Free(buffers[1]);
      return -1;
    }

    uint8_t *dst0 = buffers[0];
    uint8_t *dst1 = buffers[1];

    for (Py_ssize_t i = 0; i < len; i += 2) {
      *dst0++ = src[i];
      *dst1++ = src[i + 1];
    }
    break;

  case 8: // 4b1000 - Truncate MSByte
          // We are refering to the MSBbyte as little endian, thus we omit buf2
  case 1: // 4b1000 - Truncate LSByte
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

  uint32_t sign = (value.u << 8) & 0x80008000;
  uint32_t exponent = (value.u >> 1) & 0x7F807F80;
  uint32_t mantissa = (value.u) & 0x7F007F;
  return sign | exponent | mantissa;
}

// Helper function to reorder all floats in a bytearray
static void revert_all_floats(uint8_t *src, Py_ssize_t len) {
  uint32_t *uint_array = (uint32_t *)src;
  Py_ssize_t num_floats = len / sizeof(uint32_t);
  for (Py_ssize_t i = 0; i < num_floats; i++) {
    uint_array[i] = revert_float_bits(*(float *)&uint_array[i]);
  }
}

// Helper function to combine four buffers into a single bytearray
static uint8_t *combine_buffers(uint8_t *buf1, uint8_t *buf2, Py_ssize_t half_len,
                             int bytes_mode, int threads) {
  Py_ssize_t total_len = half_len * 2;
  uint8_t *result = NULL; // Declare result at the beginning of the function
  uint8_t *dst;
  result = PyMem_Malloc(total_len);
  dst = result;
  if (result == NULL) {
    return NULL;
  }

  switch (bytes_mode) {
  case 6: // 2b0110 - Byte Group to two different groups

    if (result == NULL) {
      return NULL;
    }

    for (Py_ssize_t i = 0; i < half_len; i++) {
      *dst++ = buf1[i];
      *dst++ = buf2[i];
    }
    break;

  case 8: // 4b1000 - Truncate MSByte
          // We are refering to the MSByte as a little endian, thus we omit buf2
  case 1: // 4b1000 - Truncate LSByte
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
    // we are not supportin this splitting bytes_mode
    return NULL;
  }
  return result;
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
  Py_buffer view;
  int bits_mode, bytes_mode, is_review, threads;
  uint8_t isPrint = 1;
  clock_t startTime, endTime, startBGTime, endBGTime, startCompBufTime[numBuf], endCompBufTime[numBuf];
  double bgTime, compBufTime[numBuf];

  if (isPrint) {
      startTime = clock();
  }

  if (!PyArg_ParseTuple(args, "y*iiii", &view, &bits_mode, &bytes_mode,
                        &is_review, &threads)) {
    return NULL;
  }

  uint8_t *buffers[] = {
    NULL, NULL
  };

  if (isPrint) {
      startBGTime = clock();
  }

  if (split_bytearray(view.buf, view.len, buffers, bits_mode, bytes_mode,
                      is_review, threads) != 0) {
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    return NULL;
  }
  if (isPrint) {
      endBGTime = clock();
      bgTime = (double)(endBGTime - startBGTime) / CLOCKS_PER_SEC;
  }
 
  ///// Compression using huffman /////////
  uint32_t chunk_size = 128 * 1024; // TBD 
  float compThreshold = 0.95; // TBD
  size_t checkThreshold = 10; // TBD

  size_t bufSize = view.len / numBuf;
  size_t maxCompressedSize = HUF_compressBound(bufSize);

  uint8_t *compressedData[] = {NULL, NULL};
  size_t numChunks = (bufSize + chunk_size -1)/ chunk_size;
  size_t* compressedChunksSize[numBuf];
  size_t bufNumChunks[numBuf];

  for (uint32_t i = 0; i < numBuf; i++) {
      compressedData[i] = PyMem_Malloc(maxCompressedSize);
      compressedChunksSize[i] = PyMem_Malloc(numChunks*sizeof(size_t));

      if (!compressedData[i] | !compressedChunksSize[i]) {
	  for (uint32_t j = 0; j < i; j++) {
              free(compressedData[j]); 
	      free(compressedChunksSize[j]);
          }    
          PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
          return NULL;
      }
  }
  
  uint8_t isBufComp[numBuf];
  size_t totalCompressedSize[numBuf];
  
  for (uint32_t i = 0; i < numBuf; i++) {
      isBufComp[i] = 1;
      bufNumChunks[i] = numChunks;
      if (isPrint) {
          startCompBufTime[i] = clock();
      }
      if (buffers[i] != NULL) {	 
          totalCompressedSize[i] = hufCompressData(buffers[i], bufSize, maxCompressedSize, compressedData[i], compressedChunksSize[i], chunk_size, compThreshold, checkThreshold);
	  if (totalCompressedSize[i] == 0) { // This buffer was not compressed
              isBufComp[i] = 0;
	      bufNumChunks[i] = 0;
	      compressedChunksSize[i] = NULL;
	  }
      }
      if (isPrint) {
          endCompBufTime[i] = clock();
          compBufTime[i] = (double)(endCompBufTime[i] - startCompBufTime[i]) / CLOCKS_PER_SEC;
      }
  }
  
  PyObject *result;
  uint8_t* resBuf[numBuf];
  size_t resBufSize[numBuf];

  if (buffers[1] != NULL) {
    for (uint32_t i = 0; i < numBuf; i++) {
        if (isBufComp[i]) { // The buffer was compressed
            resBuf[i] = compressedData[i];
	    resBufSize[i] = totalCompressedSize[i];
        }	   
        else {// The buffer was not compressed
            resBuf[i] = buffers[i];	    
	    resBufSize[i] = bufSize;
        }
    }
    uint8_t item;  
    memcpy(&item, resBuf[0], sizeof(uint8_t));
    printf ("resBuf[1] %d\n " , item);
    
    result = Py_BuildValue("y#y#y#y#y#y#", resBuf[0], resBufSize[0], resBuf[1], resBufSize[1], &isBufComp[0], 1, &isBufComp[1], 1, compressedChunksSize[0], sizeof(size_t)*bufNumChunks[0], compressedChunksSize[1], sizeof(size_t)*bufNumChunks[1]);
  } else {
    PyErr_SetString(PyExc_MemoryError, "In split16 return two buffers, still not implemented to return 1 Buffer");
    result = Py_BuildValue("y#O", buffers[0], view.len / numBuf, Py_None);
  }

  for (uint32_t i = 0; i < numBuf; i++) {
    PyMem_Free(compressedData[i]);
    PyMem_Free(compressedChunksSize[i]);
    PyMem_Free(buffers[i]);
  }
  PyBuffer_Release(&view);

  if (isPrint) {
      endTime = clock();
      double compressTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
      printf("original_size %zu \n", view.len);
      printf("BG compression time: %f seconds\n", bgTime);
      for (uint32_t i = 0; i < numBuf; i++) {
          printf("compression Buf time [%d]: %f seconds\n", i, compBufTime[i]);
          printf("totalCompressedSize[%d] %zu [%f]\n", i, totalCompressedSize[i], totalCompressedSize[i]*1.0/bufSize);
      }
      printf("compression C time: %f seconds\n", compressTime);
  }
  return result;
}

// Python callable function to combine four buffers into a single bytearray
PyObject *py_combine_dtype16(PyObject *self, PyObject *args) {
  Py_buffer view;

  int bits_mode, bytes_mode, threads;
  uint32_t numBuf = 2;
  uint32_t chunk_size = 128 * 1024; // TBD 

  if (!PyArg_ParseTuple(args, "y*iii", &view, &bits_mode,
                        &bytes_mode, &threads)) {
    return NULL;
  }

  clock_t startTime, endTime;
  uint32_t offset = 0;
  startTime = clock();
  uint8_t* bufUint8Pointer = (uint8_t *)view.buf;
  uint8_t isBufComp[numBuf];
  memcpy(&isBufComp, bufUint8Pointer + offset, numBuf * sizeof(uint8_t));
  offset += numBuf * sizeof(uint8_t);
  size_t origSize;
  memcpy(&origSize, bufUint8Pointer + offset, sizeof(size_t));
  offset += sizeof(size_t);
  size_t bufSize[numBuf];
  memcpy(&bufSize, bufUint8Pointer + offset, numBuf * sizeof(size_t));
  offset += numBuf * sizeof(size_t);
  size_t numChunks;
  memcpy(&numChunks, bufUint8Pointer + offset, sizeof(size_t));
  offset += sizeof(size_t);
  size_t compressedChunksSize[numBuf][numChunks]; 
  size_t decompressedSize[numBuf];
  uint8_t* decompressedData[] = {NULL, NULL};
  size_t offset_compressedChunksSize = offset;

  for (uint32_t i = 0; i < numBuf; i++) {
	  if (isBufComp[i]){
              decompressedData[i] = malloc(origSize/numBuf);  		  
              if (!decompressedData[i]) {
	          for (uint32_t j = 0; j < i; j++) {
                      PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
                      free(decompressedData[i]);
		  }
              }
	      printf ("offset_compressedChunksSize %zu\n", offset_compressedChunksSize); 
	      memcpy(compressedChunksSize[i], bufUint8Pointer + offset_compressedChunksSize, numChunks * sizeof(size_t));
              printf ("compressedChunksSize[1][numChunks-1] %zu\n " , compressedChunksSize[i][numChunks-1]);
	      printf ("offset %d\n", offset); 
	      decompressedSize[i] = hufDecompressData(bufUint8Pointer + offset, compressedChunksSize[i], numChunks, origSize/numBuf, decompressedData[i], chunk_size);
	      if (i < numBuf -1) {
		      offset += compressedChunksSize[i][numChunks-1];
		      offset_compressedChunksSize += numChunks * sizeof(size_t); 
	      }
	  }
          else{
                  decompressedData[i] = bufUint8Pointer + offset + numChunks * sizeof(size_t); 
		  offset += origSize/numBuf + numChunks * sizeof(size_t);
	  }
  }
  uint8_t item;  
  memcpy(&item, bufUint8Pointer + offset, sizeof(uint8_t));
  /*
  printf ("bufUint8Pointer + offset + origSize/2 %d\n " , item);
  endTime = clock();
  double compressTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
  printf("decompression C time: %f seconds\n", compressTime);
  printf ("buf len %zu\n " , view.len);
  printf ("bits_mode %d\n " , bits_mode);
  printf ("bytes_mode %d\n " , bytes_mode);
  printf ("threads %d\n " , threads);
  printf ("isBufComp[0] %d\n " , isBufComp[0]);
  printf ("isBufComp[1] %d\n " , isBufComp[1]);
  printf ("Original size %zu\n " , origSize);
  printf ("BufLen1 %zu\n " , bufSize[0]);
  printf ("BufLen2 %zu\n " , bufSize[1]);
  printf ("numChunks %zu\n " , numChunks);
  printf ("compressedChunksSize[1][0] %zu\n " , compressedChunksSize[1][0]);
  printf ("compressedChunksSize[1][1] %zu\n " , compressedChunksSize[1][1]);
  printf ("compressedChunksSize[1][numChunks-1] %zu\n " , compressedChunksSize[1][numChunks-1]);
  printf ("decompressedSize[1] %zu\n", decompressedSize[1]);
  */
  uint8_t *result = combine_buffers((uint8_t *)decompressedData[0], (uint8_t *)decompressedData[1],
                                origSize/2 , bytes_mode, threads);
  if (result == NULL) {
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    return NULL;
  }

  // Revert the reordering of all floats if needed
  if (bits_mode == 1) {
    revert_all_floats(result, origSize);
  }
  PyObject *py_result = PyByteArray_FromStringAndSize((const char *)result, origSize);
  PyMem_Free(result);
  PyBuffer_Release(&view);

  return py_result;
}
