#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include "data_manipulation_dtype16.h"
#include "data_manipulation_dtype32.h"
#include "huf.h"
#include "split_dtype_functions.h"
#include <pthread.h>
#include <sys/time.h>

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

  u_int8_t *resultBuf = malloc(*resBufSize);
  if (!resultBuf) {
    PyErr_SetString(
        PyExc_MemoryError,
        "Failed to allocate memory for result buffer in split function");
    free(resultBuf);
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
  int numBuf, bits_mode, bytes_mode, is_redata, checkThAfterPercent, threads;
  size_t origChunkSize;
  float compThreshold;
  // u_int8_t isPrint = 0;
  // clock_t startTime, endTime, startBGTime, endBGTime;
  // startTime = clock();

  if (!PyArg_ParseTuple(args, "y*y*iiiinfii", &header, &data, &numBuf,
                        &bits_mode, &bytes_mode, &is_redata, &origChunkSize,
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

  /////////// start multi Threading - Each chunk to different thread ////////////////
  for (size_t offset = 0; offset < (size_t)data.len; offset += origChunkSize) {
    size_t curOrigChunkSize = (data.len - offset > origChunkSize)
                                  ? origChunkSize
                                  : (data.len - offset);

    // Byte Grouping + Byte Ordering
    if (numBuf == 2) {
      if (split_bytearray_dtype16(data.buf + offset, curOrigChunkSize,
                                  buffers[curChunk], unCompChunksSize[curChunk],
                                  bits_mode, bytes_mode, is_redata,
                                  threads) != 0) {
        PyBuffer_Release(&data);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
      }
    } else {  // numBuf == 4
      if (split_bytearray_dtype32(data.buf + offset, curOrigChunkSize,
                                  buffers[curChunk], unCompChunksSize[curChunk],
                                  bits_mode, bytes_mode, is_redata,
                                  threads) != 0) {
        PyBuffer_Release(&data);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
      }
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

      compressedData[b][curChunk] = malloc(origChunkSize);
      if (!compressedData[b][curChunk]) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        for (int j = 0; j < numBuf; j++) {
          for (uint32_t c = 0; c < curChunk - 1; c++) {
            free(compressedData[j][c]);
          }
          for (int j = 0; j < b; j++) {
            free(compressedData[j][curChunk]);
          }
        }
        return NULL;
      }

      if (buffers[curChunk][b] != NULL) {
        if (noNeedToCompress[b] == 0) {
          compChunksSize[b][curChunk] =
              HUF_compress(compressedData[b][curChunk], origChunkSize,
                           buffers[curChunk][b], unCompChunksSize[curChunk][b]);
        } else {
          compChunksSize[b][curChunk] = 0;
        }

        if (compChunksSize[b][curChunk] != 0 &&
            (compChunksSize[b][curChunk] <
             unCompChunksSize[curChunk][b] * compThreshold)) {
          compChunksType[b][curChunk] = 1;  // Compress with Huffman
        } else {                            // the buffer was not compressed
          free(compressedData[b][curChunk]);
          compChunksSize[b][curChunk] = unCompChunksSize[curChunk][b];
          compChunksType[b][curChunk] = 0;  // not compressed
          compressedData[b][curChunk] = buffers[curChunk][b];
        }
        totalCompressedSize[b] += compChunksSize[b][curChunk];
        totalUnCompressedSize[b] += unCompChunksSize[curChunk][b];
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
     for (int b = 0; b < numBuf; b++) {
       if (buffers[c][b] != NULL) {
         free(buffers[c][b]);
       }
     }
   }


  for (uint32_t c = 0; c < numChunks; c++) {
    for (int b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] == 1) {
        free(compressedData[b][c]);
      }
    }
  }
  PyBuffer_Release(&header);
  PyBuffer_Release(&data);
  free(resultBuf);
  return result;
}


typedef struct {
    size_t chunk_id;
    int numBuf;
    int bits_mode;
    int bytes_mode;
    uint8_t **ptrCompressData;
    uint32_t (*compChunksType);         
    size_t (*compCumulativeChunksPos);  
    size_t (*compChunksLen);            
    uint8_t *resultBuf;
    uint8_t ***deCompressedDataPtr;
    size_t (*decompLen);                
    size_t origChunkSize;
    pthread_mutex_t *next_chunk_mutex;
    size_t *next_chunk;
} ChunkThreadData;

static void* process_chunk_worker(void* arg) {
    ChunkThreadData* data = (ChunkThreadData*)arg;
    size_t current_chunk;
    
    while (1) {
        // Get next chunk to process
        pthread_mutex_lock(data->next_chunk_mutex);
        current_chunk = (*data->next_chunk)++;
        pthread_mutex_unlock(data->next_chunk_mutex);
        
        if (current_chunk >= data->chunk_id) {
            break;
        }
        // Decompress each buffer for this chunk
        for (int b = 0; b < data->numBuf; b++) {
        			
            // Access 2D array [b][current_chunk]
            if (data->compChunksType[b * data->chunk_id + current_chunk] == 0) {
                data->deCompressedDataPtr[b][current_chunk] = 
                    data->ptrCompressData[b] + 
                    data->compCumulativeChunksPos[b * (data->chunk_id + 1) + current_chunk];
            } else if (data->compChunksType[b * data->chunk_id + current_chunk] == 1) {
                // Get decompLen[current_chunk][b]
                size_t decomp_length = data->decompLen[current_chunk * data->numBuf + b];
               if (data->deCompressedDataPtr != NULL) {
              }

  	       data->deCompressedDataPtr[b][current_chunk] = malloc(decomp_length);
               if (!data->deCompressedDataPtr[b][current_chunk]) {
                   pthread_exit((void*)-1);
              }
                
	       size_t decompressedSize = HUF_decompress(
			       data->deCompressedDataPtr[b][current_chunk],
			       decomp_length,
			       (void*)(data->ptrCompressData[b] + 
				       data->compCumulativeChunksPos[b * (data->chunk_id + 1) + current_chunk]),
			       data->compChunksLen[b * data->chunk_id + current_chunk]);
               free(data->deCompressedDataPtr[b][current_chunk]); 
	       if (HUF_isError(decompressedSize)) {
		       free(data->deCompressedDataPtr[b][current_chunk]);
		       pthread_exit((void*)-1);
                }
            }
        }
        
        // Combine buffers
        uint8_t *combinePtr = data->resultBuf + data->origChunkSize * current_chunk;
        if (data->numBuf == 2) {
            // Get decompLen array for current chunk
            size_t *current_decompLen = &data->decompLen[current_chunk * data->numBuf];
            
            if (combine_buffers_dtype16(
                    data->deCompressedDataPtr[0][current_chunk],
                    data->deCompressedDataPtr[1][current_chunk],
                    combinePtr,
                    current_decompLen,
                    data->bits_mode,
                    data->bytes_mode,
                    1) != 0) {
                pthread_exit((void*)-1);
            }
        } else {
            // Get decompLen array for current chunk
            size_t *current_decompLen = &data->decompLen[current_chunk * data->numBuf];
            
            if (combine_buffers_dtype32(
                    data->deCompressedDataPtr[0][current_chunk],
                    data->deCompressedDataPtr[1][current_chunk],
                    data->deCompressedDataPtr[2][current_chunk],
                    data->deCompressedDataPtr[3][current_chunk],
                    combinePtr,
                    current_decompLen,
                    data->bits_mode,
                    data->bytes_mode,
                    1) != 0) {
                pthread_exit((void*)-1);
            }
        }
    }
    
    pthread_exit(NULL);
}


// Python callable function to combine four buffers into a single bytearray
PyObject *py_combine_dtype(PyObject *self, PyObject *args) {
  Py_buffer data;
   
  
  clock_t sTime, eTime;
  sTime = clock();
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
  size_t decompLen[numChunks][numBuf];
  uint8_t ***deCompressedDataPtr = malloc(numBuf * sizeof(uint8_t **));
  if (deCompressedDataPtr == NULL) {
//     Handle error
  }
  for(int i = 0; i < numBuf; i++) {
    deCompressedDataPtr[i] = malloc(numChunks * sizeof(uint8_t *));
    if (deCompressedDataPtr[i] == NULL) {
        // Handle error
    }
    for(int j = 0; j < numChunks; j++) {
        deCompressedDataPtr[i][j] = NULL;  
    }
  }


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
  for (uint32_t c = 0; c < numChunks; c++) {
    for (int b = 0; b < numBuf; b++) {
      compCumulativeChunksPos[b][c + 1] = cumulativeChunksSize[b][c];
      compChunksLen[b][c] =
          compCumulativeChunksPos[b][c + 1] - compCumulativeChunksPos[b][c];
    }
  }
  for (size_t c = 0; c < numChunks; c++) {
    for (int b = 0; b < numBuf; b++) {
      if (compChunksType[b][c] == 0) {  // no compression is needed
      } else {
        if (compChunksType[b][c] == 1) {  // open with Huffman compression
        } else {
          PyErr_SetString(
              PyExc_MemoryError,
              "Compress Type is not correct in Decompression function");
          return NULL;
        }
      }
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

  resultBuf = malloc(origSize);
  if (!resultBuf) {
    PyErr_SetString(
        PyExc_MemoryError,
        "Failed to allocate memory for result buffer in split function");
    free(resultBuf);
    return NULL;
  }
  eTime = clock();
  double metadataTime = (double)(eTime - sTime) / CLOCKS_PER_SEC;
//  printf ("metadataTime %f\n", metadataTime);

//  clock_t startTime, endTime;
//  startTime = clock();
  struct timeval startTimeReal, endTimeReal;
  gettimeofday(&startTimeReal, NULL);



  ////////////// Multi threading /////////////////////////////
   pthread_t *thread_handles = NULL;
   ChunkThreadData *thread_data = NULL;
   pthread_mutex_t next_chunk_mutex = PTHREAD_MUTEX_INITIALIZER;
   size_t next_chunk = 0;
   int mutex_initialized = 1;
   thread_handles = malloc(threads * sizeof(pthread_t));
   thread_data = malloc(threads * sizeof(ChunkThreadData));
   if (!thread_handles || !thread_data) {
       PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
       goto cleanup_threads;
   }

   // Create threads
   for (int i = 0; i < threads; i++) {
       thread_data[i] = (ChunkThreadData){
           .chunk_id = numChunks,
           .numBuf = numBuf,
           .bits_mode = bits_mode,
           .bytes_mode = bytes_mode,
           .ptrCompressData = ptrCompressData,
           .compChunksType = (uint32_t *)compChunksType,
           .compCumulativeChunksPos = (size_t *)compCumulativeChunksPos,
           .compChunksLen = (size_t *)compChunksLen,
           .resultBuf = resultBuf,
           .deCompressedDataPtr = deCompressedDataPtr,
           .decompLen = (size_t *)decompLen,
           .origChunkSize = origChunkSize,
           .next_chunk_mutex = &next_chunk_mutex,
           .next_chunk = &next_chunk

       };
       if (pthread_create(&thread_handles[i], NULL, process_chunk_worker, &thread_data[i]) != 0) {
           PyErr_SetString(PyExc_RuntimeError, "Failed to create thread");
           goto cleanup_threads;
       }
   }

   // Wait for all threads
   for (int i = 0; i < threads; i++) {
       void *thread_result;
       pthread_join(thread_handles[i], &thread_result);
       if (thread_result != NULL) {
           PyErr_SetString(PyExc_RuntimeError, "Thread processing failed");
           goto cleanup_threads;
       }
   }

   free(thread_handles);
   free(thread_data);
   pthread_mutex_destroy(&next_chunk_mutex);
   mutex_initialized = 0;
   goto continue_processing;

  cleanup_threads:
      if (thread_handles) free(thread_handles);
      if (thread_data) free(thread_data);
      if (mutex_initialized) pthread_mutex_destroy(&next_chunk_mutex);
      return NULL;

  ////////////// Finish Multi threading /////////////////////////////
  PyObject *py_result;  // Move declaration before label
  continue_processing:
//   endTime = clock();
//   double decompressTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
   gettimeofday(&endTimeReal, NULL);
   double decompressTimeReal = (endTimeReal.tv_sec - startTimeReal.tv_sec) + 
                            (endTimeReal.tv_usec - startTimeReal.tv_usec) / 1e6;
//   printf ("thread decompressTime %f\n", decompressTime);
//   printf("Real thread time: %f seconds\n", decompressTimeReal);

  clock_t sT, eT;
  sT = clock();

  Py_buffer view; // create buffer to avoid copy
  PyBuffer_FillInfo(&view, NULL, resultBuf, origSize, 0, PyBUF_WRITABLE);
  py_result = PyMemoryView_FromBuffer(&view);
  eT = clock();
  double resultTime = (double)(eT - sT) / CLOCKS_PER_SEC;
//  printf ("resultTime %f\n", resultTime);

  sT = clock();
  for(int b = 0; b < numBuf; b++) {
        free(deCompressedDataPtr[b]);
  }
  eT = clock();
  double freeTime = (double)(eT - sT) / CLOCKS_PER_SEC;
//  printf ("free %f\n", freeTime);


//  free(resultBuf);
//  PyBuffer_Release(&data);
  return py_result;
}
