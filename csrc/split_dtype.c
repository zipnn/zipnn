#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include "data_manipulation_dtype16.h"
#include "data_manipulation_dtype32.h"
#include "zstd.h"
#include <zstd_errors.h>
#include "split_dtype_functions.h"
#include "methods_enums.h"

// Huffman Function declaration //
size_t HUF_decompress(void* dst, size_t dstSize, const void* cSrc, size_t cSrcSize);
size_t HUF_compress(void* dst, size_t dstCapacity, const void* src, size_t srcSize);
unsigned HUF_isError(size_t code);
const char* HUF_getErrorName(size_t code);

// FSE Function declaration //
size_t FSE_decompress(void* dst, size_t dstSize, const void* cSrc, size_t cSrcSize);
size_t FSE_compress(void* dst, size_t dstCapacity, const void* src, size_t srcSize);
unsigned FSE_isError(size_t code);
const char* FSE_getErrorName(size_t code);

// ZSTD initialzation ////
ZSTD_CCtx* cctx;  
ZSTD_DCtx* dctx; 

void initialize_zstd_contexts() {
    cctx = ZSTD_createCCtx();
    if (cctx == NULL) {
        fprintf(stderr, "Failed to create ZSTD compression context\n");
        exit(1);  // Or handle error appropriately
    }

    dctx = ZSTD_createDCtx();
    if (dctx == NULL) {
        fprintf(stderr, "Failed to create ZSTD decompression context\n");
        ZSTD_freeCCtx(cctx); 
        exit(1);  
    }
}

void cleanup_zstd_contexts() {
    // Free the contexts
    if (cctx != NULL) ZSTD_freeCCtx(cctx);
    if (dctx != NULL) ZSTD_freeDCtx(dctx);
}

////  Helper Functions ///////
u_int8_t *prepare_split_results(size_t header_len, size_t numBuf,
                                size_t numChunks, u_int8_t *header,
                                u_int8_t *compressedData[numBuf][numChunks],
                                uint32_t compChunksSize[numBuf][numChunks],
                                const u_int8_t compChunksType[numBuf][numChunks],
                                const size_t cumulativeChunksSize[numBuf][numChunks],
                                const size_t *totalCompressedSize,
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

  for (size_t b = 0; b < numBuf; b++) {
    for (uint32_t c = 0; c < numChunks; c++) {
      memcpy(resultBuf + offset, compressedData[b][c], compChunksSize[b][c]);
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
//     1 - /ereorder of the exponent (eponent, sign_bit, mantissa)
// bytes_mode:
//     [we are referring to the bytes order as first 2bits refer to the MSByte
//     and the second two bits to the LSByte] 2b [MSB Byte],2b[LSB Byte] 0 -
//     truncate this byte 1 or 2 - a group of bytes 4b0110 [6] - bytegroup to
//     two groups 4b0001 [1] - truncate the MSByte 4b1000 [8] - truncate the
//     LSByte
// is_review:
//     Even if you have the Byte mode, you can change it if needed.
//     0 - No review, take the bit_mode and byte_mode
//     1 - the function can change the Bytes_mode

PyObject *py_split_dtype(PyObject *self, PyObject *args) {
  Py_buffer header, data;
  int numBuf, bits_mode, bytes_mode, method, is_review, checkThAfterPercent, threads;
  size_t origChunkSize;
  float compThreshold;
  // u_int8_t isPrint = 0;
  // clock_t startTime, endTime, startBGTime, endBGTime;
  // startTime = clock();

  if (!PyArg_ParseTuple(args, "y*y*iiiiinfii", &header, &data, &numBuf,
                        &bits_mode, &bytes_mode, &method, &is_review, &origChunkSize,
                        &compThreshold, &checkThAfterPercent, &threads)) {
    return NULL;
  }

  // Byte Group per chunk, Compress per bufChunk
  size_t numChunks = (data.len + origChunkSize - 1) / origChunkSize;
  u_int8_t *buffers[numChunks][numBuf];
  size_t curChunk = 0;

  u_int8_t *compressedData[numBuf][numChunks];
  uint32_t compChunksSize[numBuf][numChunks];
  size_t unCompChunksSize[numChunks][numBuf];
  size_t totalCompressedSize[numBuf];
  size_t totalUnCompressedSize[numBuf];
  u_int8_t compChunksType[numBuf][numChunks];
  size_t cumulativeChunksSize[numBuf][numChunks];
  u_int8_t isThCheck[numBuf];
  u_int8_t noNeedToCompress[numBuf];
  uint32_t checkCompTh =
      (uint32_t)ceil((double)numChunks / checkThAfterPercent);
  int chunk_methods[numBuf];
  int split_result;
  // if (isPrint) {
  //     startBGTime = clock();
  //  }

  // initialize:
  for (int b = 0; b < numBuf; b++) {
    totalCompressedSize[b] = 0;
    totalUnCompressedSize[b] = 0;
    isThCheck[b] = 0;
    noNeedToCompress[b] = 0;
  }

  /////////// start multi Threading - Each chunk to different thread
  //////////////////
  for (size_t offset = 0; offset < (size_t)data.len; offset += origChunkSize) {
    size_t curOrigChunkSize = (data.len - offset > origChunkSize)
                                  ? origChunkSize
                                  : (data.len - offset);

    // Byte Grouping + Byte Ordering
    if (numBuf == 2) {
      split_result = split_bytearray_dtype16(data.buf + offset, curOrigChunkSize,
                                  buffers[curChunk], unCompChunksSize[curChunk],
                                  bits_mode, bytes_mode, method, chunk_methods, is_review,
                                  threads);
    } else {  // numBuf == 4
      split_result = split_bytearray_dtype32(data.buf + offset, curOrigChunkSize,
                                  buffers[curChunk], unCompChunksSize[curChunk],
                                  bits_mode, bytes_mode, method, chunk_methods, is_review, threads);
                                  
    }
    
    if (split_result == -1) {
        PyBuffer_Release(&data);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
      }


    //    if (isPrint) {
    //      endBGTime = clock();
    //      bgTime = (double)(endBGTime - startBGTime) / CLOCKS_PER_SEC;
    //    }
    // Compression on each Buf

    for (int b = 0; b < numBuf; b++) {
      if (isThCheck[b] == 0 &&
          curChunk >=
              checkCompTh) {  // check that we really need to compress this buf
        isThCheck[b] = 1;
        if (totalCompressedSize[b] * 1.0 >
            totalUnCompressedSize[b] * compThreshold) {
          noNeedToCompress[b] = 1;
        }
      }

      compressedData[b][curChunk] = PyMem_Malloc(origChunkSize);
      if (!compressedData[b][curChunk]) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        for (int j = 0; j < numBuf; j++) {
          for (uint32_t c = 0; c < curChunk - 1; c++) {
            PyMem_Free(compressedData[j][c]);
          }
          for (int j = 0; j < b; j++) {
            PyMem_Free(compressedData[j][curChunk]);
          }
        }
        return NULL;
      }

      compChunksType[b][curChunk] = chunk_methods[b]; // HUFFMAN FSE ZSTD
      if (buffers[curChunk][b] != NULL) {
       if (noNeedToCompress[b] == 0) {
         switch (compChunksType[b][curChunk]) {
           case HUFFMAN: {
             compChunksSize[b][curChunk] = HUF_compress(
                 compressedData[b][curChunk], origChunkSize,
                 buffers[curChunk][b], unCompChunksSize[curChunk][b]);
             if (HUF_isError(compChunksSize[b][curChunk])) {
                 HUF_getErrorName(compChunksSize[b][curChunk]);
                 PyErr_SetString(PyExc_MemoryError,
                                 "Huffman compression returned an error");
                 return NULL;
             }
             break;
   	   }
 
           case ZSTD: {
 	    size_t zstd_levels = 1;
             compChunksSize[b][curChunk] = ZSTD_compress(
                 compressedData[b][curChunk], origChunkSize,
                 buffers[curChunk][b], unCompChunksSize[curChunk][b], zstd_levels);
             if (ZSTD_isError(compChunksSize[b][curChunk])) {
                 ZSTD_getErrorName(compChunksSize[b][curChunk]);
                 PyErr_SetString(PyExc_MemoryError,
                                 "ZSTD compression returned an error");
                 return NULL;
             }
             break;
           }
 
           case FSE: {
             compChunksSize[b][curChunk] = FSE_compress(
                 compressedData[b][curChunk], origChunkSize,
                 buffers[curChunk][b], unCompChunksSize[curChunk][b]);
             if (FSE_isError(compChunksSize[b][curChunk])) {
                 FSE_getErrorName(compChunksSize[b][curChunk]);
                 PyErr_SetString(PyExc_MemoryError,
                                 "FSE compression returned an error");
                 return NULL;
             }
             break;
   	   }
            
           case TRUNCATE: {
             compChunksSize[b][curChunk] = 0;
             compressedData[b][curChunk] = NULL;
           }
             break;
 
           default: {
             fprintf(stderr, "Unknown compression type for chunk %d\n", curChunk);
             break;
           }
         }
       } else {
        compChunksSize[b][curChunk] = 0;
        }
      }

      if (chunk_methods[b] != TRUNCATE) {
        if (compChunksSize[b][curChunk] != 0 &&
         (compChunksSize[b][curChunk] <
            unCompChunksSize[curChunk][b] * compThreshold)) {
        } else {                            // the buffer was not compressed
          PyMem_Free(compressedData[b][curChunk]);
          compChunksSize[b][curChunk] = unCompChunksSize[curChunk][b];
          compChunksType[b][curChunk] = ORIGINAL;  // not compressed
          compressedData[b][curChunk] = buffers[curChunk][b];
       }
        totalCompressedSize[b] += compChunksSize[b][curChunk];
        totalUnCompressedSize[b] += unCompChunksSize[curChunk][b];
        cumulativeChunksSize[b][curChunk] = totalCompressedSize[b];
      }
      else { // TRUNCATE
        if (curChunk > 0) {
          cumulativeChunksSize[b][curChunk] = cumulativeChunksSize[b][curChunk-1];
	}
	else{
          cumulativeChunksSize[b][curChunk] = 0;
	}
      }  
    }// end for loop -> compression
    curChunk++;
  }  // end for loop - chunk

  ////////////// The end of multi Threading part 1
  /////////////////////////////////

  // endTime = clock();
  // double compressTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

  PyObject *result;
  u_int8_t *resultBuf;
  size_t resBufSize;
  size_t is_print_compression_size = 1;
  if (is_print_compression_size) {
    for (int b = 0; b < numBuf; b++) {
      printf("Group[%d] compression %.6f \n", b, totalCompressedSize[b] * 1.0/ (data.len/numBuf));	    
    }       
  }



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
    for (int b = 0; b < numBuf; b++) {
      if (buffers[c][b] != NULL) {
        PyMem_Free(buffers[c][b]);
      }
    }
  }

  for (uint32_t c = 0; c < numChunks; c++) {
    for (int b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] == HUFFMAN) {
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
PyObject *py_combine_dtype(PyObject *self, PyObject *args) {
  Py_buffer data;

  int numBuf, bits_mode, bytes_mode, threads;
  size_t origChunkSize, origSize;

  if (!PyArg_ParseTuple(args, "y*iiinni", &data, &numBuf, &bits_mode,
                        &bytes_mode, &origChunkSize, &origSize, &threads)) {
    return NULL;
  }
  size_t numChunks = (origSize + origChunkSize - 1) / origChunkSize;
  uint32_t bufRatio[numChunks][numBuf];
  uint32_t unCompChunkSize[numChunks][numBuf];
  uint32_t oneBufRatio[numBuf];
  uint32_t oneUnCompChunkSize[numBuf];

  if (1) {  // TBD when support auto byte_reorder
    if (numBuf == 2) {
      if (buffer_ratio_dtype16(bytes_mode, oneBufRatio) == -1) {
        PyErr_SetString(PyExc_MemoryError, "Failed to calculate bufffer ratio");
        return NULL; //
      }
    } else {  // numBuf == 4
      if (buffer_ratio_dtype32(bytes_mode, oneBufRatio) == -1) {
        PyErr_SetString(PyExc_MemoryError, "Failed to calculate bufffer ratio");
        return NULL;
      }
    }
    for (int b = 0; b < numBuf; b++) {
      oneBufRatio[b] = numBuf;
      oneUnCompChunkSize[b] = origChunkSize / oneBufRatio[b];
    }
    for (uint32_t c = 0; c < numChunks; c++) {
      for (int b = 0; b < numBuf; b++) {
        unCompChunkSize[c][b] = oneUnCompChunkSize[b];
      }
    }
  } else {
    // TBD when support dynamic byte_reorder
  }

  u_int8_t *ptrChunksType = (u_int8_t *)data.buf;
  size_t *ptrChunksCumulative = (size_t *)(ptrChunksType + numBuf * numChunks);
  u_int8_t *ptrCompressData[numBuf];
  ptrCompressData[0] = (u_int8_t *)(ptrChunksCumulative + numBuf * numChunks);
  size_t cumulativeChunksSize[numBuf][numChunks];
  uint32_t compChunksType[numBuf][numChunks];
  size_t compCumulativeChunksPos[numBuf][numChunks + 1];
  size_t compChunksLen[numBuf][numChunks];
  u_int8_t *resultBuf = NULL;
  u_int8_t *deCompressedData[numBuf][numChunks];
  size_t decompLen[numChunks][numBuf];
  size_t decompressedSize;
  int is_print_method = 1;
  // clock_t startTime, endTime;
  // startTime = clock();

  // Preparation for decompression
  for (int b = 0; b < numBuf; b++) {
    compCumulativeChunksPos[b][0] = 0;
    compCumulativeChunksPos[b][0] = 0;
  }

  for (int b = 0; b < numBuf; b++) {
    for (uint32_t c = 0; c < numChunks; c++) {
      compChunksType[b][c] = (*ptrChunksType++);
      cumulativeChunksSize[b][c] = (*ptrChunksCumulative++);
    }
  }
  
  if (is_print_method) {
    for (uint32_t c = 0; c < numChunks; c++) {
      for (int b = 0; b < numBuf; b++) {
        printf("Compression method Chunk [%zu] Group [%d] %s\n", c, b, getEnumName(compChunksType[b][c]));
      }	      
    }	    
  }
  for (uint32_t c = 0; c < numChunks; c++) {
    for (int b = 0; b < numBuf; b++) {
      compCumulativeChunksPos[b][c + 1] = cumulativeChunksSize[b][c];
      compChunksLen[b][c] =
          compCumulativeChunksPos[b][c + 1] - compCumulativeChunksPos[b][c];
    }
  }
  
  for (int b = 1; b < numBuf; b++) {
    ptrCompressData[b] =
        ptrCompressData[b - 1] + cumulativeChunksSize[b - 1][numChunks - 1];
  }
  for (uint32_t c = 0; c < numChunks; c++) {
    if (c < numChunks - 1) {
      for (int b = 0; b < numBuf; b++) {
        decompLen[c][b] = unCompChunkSize[c][b];
      }
    } else {
      size_t oneChunkSize = 0;
      for (int b = 0; b < numBuf; b++) {
        oneChunkSize += unCompChunkSize[0][b];
      }

      size_t lastDecompLen =
          (origSize - oneChunkSize * (numChunks - 1)) / numBuf;
      int remainder = (origSize - oneChunkSize * (numChunks - 1)) % numBuf;
      for (int b = 0; b < numBuf; b++) {
        if (b < remainder) {
          decompLen[c][b] = lastDecompLen + 1;
        } else {
          decompLen[c][b] = lastDecompLen;
        }
      }
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
    for (int b = 0; b < numBuf; b++) {

      switch (compChunksType[b][c]) {
        case ORIGINAL:  // No compression needed
	  deCompressedData[b][c] = ptrCompressData[b] + compCumulativeChunksPos[b][c];
          break;

        case HUFFMAN:  // Decompress using Huffman
          deCompressedData[b][c] = PyMem_Malloc(decompLen[c][b]);
          if (deCompressedData[b][c] == NULL) {
            PyErr_SetString(
                PyExc_MemoryError,
                "Failed to allocate memory - Function during decompression"
            );
            PyMem_Free(deCompressedData[b][c]);
            return NULL;
	  }
          // Add logic for Huffman decompression here
          decompressedSize = HUF_decompress(
          deCompressedData[b][c], decompLen[c][b],
          (void *)(ptrCompressData[b] + compCumulativeChunksPos[b][c]),
          compChunksLen[b][c]);

          if (HUF_isError(decompressedSize)) {
            HUF_getErrorName(decompressedSize);
            PyErr_SetString(PyExc_MemoryError,
			     "Hufman decompression returned an error");
            return NULL;
	  }

	  if (decompressedSize != decompLen[c][b]) {
            PyErr_SetString(
              PyExc_MemoryError,
             "decompressedSize is not equal the expected decompressedSize");
            return NULL;
	  }
          break;

        case ZSTD:  // Decompress using ZSTD
          deCompressedData[b][c] = PyMem_Malloc(decompLen[c][b]);
          if (deCompressedData[b][c] == NULL) {
            PyErr_SetString(
                PyExc_MemoryError,
                "Failed to allocate memory - Function during decompression"
            );
            PyMem_Free(deCompressedData[b][c]);
            return NULL;
	  }
          decompressedSize = ZSTD_decompress(
          deCompressedData[b][c], decompLen[c][b],
          (void *)(ptrCompressData[b] + compCumulativeChunksPos[b][c]),
          compChunksLen[b][c]);

          if (ZSTD_isError(decompressedSize)) {
            ZSTD_getErrorName(decompressedSize);
            PyErr_SetString(PyExc_MemoryError,
			     "Hufman decompression returned an error");
            return NULL;
	  }

	  if (decompressedSize != decompLen[c][b]) {
            PyErr_SetString(
              PyExc_MemoryError,
             "decompressedSize is not equal the expected decompressedSize");
            return NULL;
	  }
          break;

        case FSE:  // Decompress using FSE
          deCompressedData[b][c] = PyMem_Malloc(decompLen[c][b]);
          if (deCompressedData[b][c] == NULL) {
            PyErr_SetString(
                PyExc_MemoryError,
                "Failed to allocate memory - Function during decompression"
            );
            PyMem_Free(deCompressedData[b][c]);
            return NULL;
	  }
          // Add logic for Huffman decompression here
          decompressedSize = FSE_decompress(
          deCompressedData[b][c], decompLen[c][b],
          (void *)(ptrCompressData[b] + compCumulativeChunksPos[b][c]),
          compChunksLen[b][c]);

          if (FSE_isError(decompressedSize)) {
            FSE_getErrorName(decompressedSize);
            PyErr_SetString(PyExc_MemoryError,
			     "FSE decompression returned an error");
            return NULL;
	  }

	  if (decompressedSize != decompLen[c][b]) {
            PyErr_SetString(
              PyExc_MemoryError,
             "decompressedSize is not equal the expected decompressedSize");
            return NULL;
	  }
          break;


        case TRUNCATE:  // Truncate decompression
	  deCompressedData[b][c] = (u_int8_t*)PyMem_Calloc(decompLen[c][b], sizeof(u_int8_t));
          if (deCompressedData[b][c] == NULL) {
            PyErr_SetString(
                PyExc_MemoryError,
                "Failed to allocate memory - Function during decompression"
            );
            PyMem_Free(deCompressedData[b][c]);
            return NULL;
	  }
          break;
 
        default:  
	  PyErr_SetString(
            PyExc_ValueError,
            "Unknown compression type during decompression"
          );
          return NULL;
       }
    }
    // Combine
    u_int8_t *combinePtr = resultBuf + origChunkSize * c;
    if (numBuf == 2) {
      if (combine_buffers_dtype16(
              deCompressedData[0][c], deCompressedData[1][c], combinePtr,
              decompLen[c], bits_mode, bytes_mode, threads) != 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to combine dtype16");
        return NULL;
      }
    } else {  // Assume numBuf == 4
      if (combine_buffers_dtype32(
              deCompressedData[0][c], deCompressedData[1][c],
              deCompressedData[2][c], deCompressedData[3][c], combinePtr,
              decompLen[c], bits_mode, bytes_mode, threads) != 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to combine dtype16");
        return NULL;
      }
    }
  }
  ////////////// Finish Multi threading /////////////////////////////

  PyObject *py_result =
      PyByteArray_FromStringAndSize((const char *)resultBuf, origSize);

  for (size_t c = 0; c < numChunks; c++) {
    for (int b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] > 0) {
        PyMem_Free(deCompressedData[b][c]);
      }
    }
  }

  PyMem_Free(resultBuf);
  PyBuffer_Release(&data);
  return py_result;
}
