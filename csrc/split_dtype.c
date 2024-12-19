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

////  Helper Functions //////
/*
u_int8_t *prepare_split_results(size_t header_len, size_t numBuf,
                                size_t numChunks, u_int8_t *header,
                                u_int8_t ***compressedData,
                                uint32_t **compChunksSize, 
                                const u_int8_t **compChunksType,
                                const size_t **cumulativeChunksSize,  // Ignore this
                                const size_t *totalCompressedSize,
                                size_t *resBufSize) {
    
    *resBufSize = header_len;
    
    // Calculate sizes
    size_t compChunksTypeLen = numBuf * numChunks * sizeof(u_int8_t);
    size_t cumulativeChunksSizeLen = numBuf * numChunks * sizeof(size_t);
    
    *resBufSize += compChunksTypeLen;
    *resBufSize += cumulativeChunksSizeLen;
    
    for (size_t b = 0; b < numBuf; b++) {
        *resBufSize += totalCompressedSize[b];
        printf("Debug 3.%zu: totalCompressedSize[%zu] = %zu\n", b, b, totalCompressedSize[b]);
    }
    
    memcpy(&header[24], resBufSize, sizeof(size_t));
    
    u_int8_t *resultBuf = (u_int8_t *)malloc(*resBufSize);
    if (!resultBuf) {
        PyErr_SetString(PyExc_MemoryError,
            "Failed to allocate memory for result buffer in split function");
        return NULL;
    }
    
    size_t offset = 0;
    memcpy(resultBuf + offset, header, header_len);
    offset += header_len;
    
    // Copy compChunksType data
    u_int8_t *typePtr = resultBuf + offset;
    for (size_t b = 0; b < numBuf; b++) {
        if (!compChunksType[b]) {
            printf("Error: compChunksType[%zu] is NULL\n", b);
            free(resultBuf);
            return NULL;
        }
        memcpy(typePtr + b * numChunks, compChunksType[b], numChunks * sizeof(u_int8_t));
    }
    offset += compChunksTypeLen;
    
    // Calculate and write cumulative sizes
    size_t *sizePtr = (size_t *)(resultBuf + offset);
    for (size_t b = 0; b < numBuf; b++) {
        size_t cumulative = 0;
        for (size_t c = 0; c < numChunks; c++) {
            cumulative += compChunksSize[b][c];
            sizePtr[b * numChunks + c] = cumulative;
        }
    }
    offset += cumulativeChunksSizeLen;
    
    // Copy compressed data chunks
    for (size_t b = 0; b < numBuf; b++) {
        if (!compressedData[b]) {
            printf("Error: compressedData[%zu] is NULL\n", b);
            continue;
        }
        for (uint32_t c = 0; c < numChunks; c++) {
            if (compressedData[b][c] && compChunksSize[b][c] > 0) {
                memcpy(resultBuf + offset, compressedData[b][c], compChunksSize[b][c]);
                free(compressedData[b][c]);
                offset += compChunksSize[b][c];
            }
        }
    }
    
    return resultBuf;
}


#include <pthread.h>

// Structure to hold thread data for compressed data copying
typedef struct {
    size_t b;
    size_t numChunks;
    u_int8_t ***compressedData;
    uint32_t **compChunksSize;
    u_int8_t *resultBuf;
    size_t startOffset;
    size_t *currentOffset;
    pthread_mutex_t *offsetMutex;
} CompressedDataCopyArgs;

// Thread function for copying compressed data
void* copy_compressed_data(void* arg) {
    CompressedDataCopyArgs *args = (CompressedDataCopyArgs*)arg;
    size_t localOffset = args->startOffset;
    
    if (!args->compressedData[args->b]) {
        return NULL;
    }

    for (uint32_t c = 0; c < args->numChunks; c++) {
        if (args->compressedData[args->b][c] && args->compChunksSize[args->b][c] > 0) {
            memcpy(args->resultBuf + localOffset, 
                   args->compressedData[args->b][c], 
                   args->compChunksSize[args->b][c]);
            free(args->compressedData[args->b][c]);
            localOffset += args->compChunksSize[args->b][c];
        }
    }

    // Update global offset
    pthread_mutex_lock(args->offsetMutex);
    *args->currentOffset = localOffset;
    pthread_mutex_unlock(args->offsetMutex);

    return NULL;
}
*/

/*
// Structure definition needs to be before functions
typedef struct {
    size_t b;
    size_t numChunks;
    u_int8_t ***compressedData;
    uint32_t **compChunksSize;
    u_int8_t *resultBuf;
    size_t startOffset;
    size_t *currentOffset;
    pthread_mutex_t *offsetMutex;
} CompressedDataCopyArgs;

// Thread function definition
void* copy_compressed_data(void* arg) {
    CompressedDataCopyArgs *args = (CompressedDataCopyArgs*)arg;
    size_t localOffset = args->startOffset;
    
    if (!args->compressedData[args->b]) {
        return NULL;
    }

    for (uint32_t c = 0; c < args->numChunks; c++) {
        if (args->compressedData[args->b][c] && args->compChunksSize[args->b][c] > 0) {
            memcpy(args->resultBuf + localOffset, 
                   args->compressedData[args->b][c], 
                   args->compChunksSize[args->b][c]);
            free(args->compressedData[args->b][c]);
            localOffset += args->compChunksSize[args->b][c];
        }
    }

    pthread_mutex_lock(args->offsetMutex);
    *args->currentOffset = localOffset;
    pthread_mutex_unlock(args->offsetMutex);

    return NULL;
}

u_int8_t *prepare_split_results(size_t header_len, size_t numBuf,
                               size_t numChunks, u_int8_t *header,
                               u_int8_t ***compressedData,
                               uint32_t **compChunksSize,
                               const u_int8_t **compChunksType,
                               const size_t **cumulativeChunksSize,
                               const size_t *totalCompressedSize,
                               size_t *resBufSize) {
    // Rest of the implementation remains the same
    *resBufSize = header_len;
    size_t compChunksTypeLen = numBuf * numChunks * sizeof(u_int8_t);
    size_t cumulativeChunksSizeLen = numBuf * numChunks * sizeof(size_t);
    *resBufSize += compChunksTypeLen + cumulativeChunksSizeLen;
    
    for (size_t b = 0; b < numBuf; b++) {
        *resBufSize += totalCompressedSize[b];
    }

    memcpy(&header[24], resBufSize, sizeof(size_t));
    u_int8_t *resultBuf = (u_int8_t *)malloc(*resBufSize);
    if (!resultBuf) {
        PyErr_SetString(PyExc_MemoryError,
            "Failed to allocate memory for result buffer in split function");
        return NULL;
    }

    memcpy(resultBuf, header, header_len);
    size_t offset = header_len;

    for (size_t b = 0; b < numBuf; b++) {
        if (!compChunksType[b]) {
            free(resultBuf);
            return NULL;
        }
        memcpy(resultBuf + offset + b * numChunks, 
               compChunksType[b], 
               numChunks * sizeof(u_int8_t));
    }
    offset += compChunksTypeLen;

    size_t *sizePtr = (size_t *)(resultBuf + offset);
    #pragma omp parallel for if(numBuf > 1)
    for (size_t b = 0; b < numBuf; b++) {
        size_t cumulative = 0;
        for (size_t c = 0; c < numChunks; c++) {
            cumulative += compChunksSize[b][c];
            sizePtr[b * numChunks + c] = cumulative;
        }
    }
    offset += cumulativeChunksSizeLen;

    pthread_t threads[numBuf];
    CompressedDataCopyArgs thread_args[numBuf];
    pthread_mutex_t offsetMutex = PTHREAD_MUTEX_INITIALIZER;
    size_t currentOffset = offset;

    for (size_t b = 0; b < numBuf; b++) {
        thread_args[b].b = b;
        thread_args[b].numChunks = numChunks;
        thread_args[b].compressedData = compressedData;
        thread_args[b].compChunksSize = compChunksSize;
        thread_args[b].resultBuf = resultBuf;
        thread_args[b].startOffset = offset;
        thread_args[b].currentOffset = &currentOffset;
        thread_args[b].offsetMutex = &offsetMutex;

        if (pthread_create(&threads[b], NULL, copy_compressed_data, &thread_args[b]) != 0) {
            for (size_t i = 0; i < b; i++) {
                pthread_join(threads[i], NULL);
            }
            free(resultBuf);
            return NULL;
        }
        
        for (uint32_t c = 0; c < numChunks; c++) {
            offset += compChunksSize[b][c];
        }
    }

    for (size_t b = 0; b < numBuf; b++) {
        pthread_join(threads[b], NULL);
    }

    pthread_mutex_destroy(&offsetMutex);
    return resultBuf;
}
*/

struct CompressedDataCopyArgs {
    size_t b;                   // buffer index
    size_t thread_id;          // thread identifier
    size_t start_chunk;        // starting chunk for this thread
    size_t end_chunk;          // ending chunk for this thread
    size_t numChunks;
    u_int8_t ***compressedData;
    uint32_t **compChunksSize;
    u_int8_t *resultBuf;
    size_t baseOffset;         // base offset for this buffer
    size_t *threadOffsets;     // array to store offsets for each thread
};

void* copy_compressed_data(void* arg) {
    struct CompressedDataCopyArgs *args = (struct CompressedDataCopyArgs*)arg;
    size_t localOffset = args->baseOffset;
    
    // Calculate initial offset
    for (size_t c = 0; c < args->start_chunk; c++) {
        if (args->compChunksSize[args->b][c] > 0) {
            localOffset += args->compChunksSize[args->b][c];
        }
    }

    // Process assigned chunks
    for (size_t c = args->start_chunk; c < args->end_chunk && c < args->numChunks; c++) {
        if (args->compressedData[args->b][c] && args->compChunksSize[args->b][c] > 0) {
            memcpy(args->resultBuf + localOffset, 
                   args->compressedData[args->b][c], 
                   args->compChunksSize[args->b][c]);
            free(args->compressedData[args->b][c]);
            localOffset += args->compChunksSize[args->b][c];
        }
    }
    
    // Store final offset
    args->threadOffsets[args->thread_id] = localOffset - args->baseOffset;
    return NULL;
}

u_int8_t *prepare_split_results(size_t header_len, size_t numBuf,
                               size_t numChunks, u_int8_t *header,
                               u_int8_t ***compressedData,
                               uint32_t **compChunksSize,
                               const u_int8_t **compChunksType,
                               const size_t **cumulativeChunksSize,
                               const size_t *totalCompressedSize,
                               size_t *resBufSize) {
    *resBufSize = header_len;
    size_t compChunksTypeLen = numBuf * numChunks * sizeof(u_int8_t);
    size_t cumulativeChunksSizeLen = numBuf * numChunks * sizeof(size_t);
    *resBufSize += compChunksTypeLen + cumulativeChunksSizeLen;
    
    for (size_t b = 0; b < numBuf; b++) {
        *resBufSize += totalCompressedSize[b];
    }

    // Allocate result buffer
    memcpy(&header[24], resBufSize, sizeof(size_t));
    u_int8_t *resultBuf = (u_int8_t *)malloc(*resBufSize);
    if (!resultBuf) {
        PyErr_SetString(PyExc_MemoryError,
            "Failed to allocate memory for result buffer in split function");
        return NULL;
    }

    // Copy header and type data
    size_t offset = 0;
    memcpy(resultBuf, header, header_len);
    offset += header_len;

    // Copy chunk types in parallel
    #pragma omp parallel for if(numBuf > 1)
    for (size_t b = 0; b < numBuf; b++) {
        if (compChunksType[b]) {
            memcpy(resultBuf + offset + b * numChunks, 
                   compChunksType[b], 
                   numChunks * sizeof(u_int8_t));
        }
    }
    offset += compChunksTypeLen;

    // Calculate cumulative sizes in parallel
    size_t *sizePtr = (size_t *)(resultBuf + offset);
    #pragma omp parallel for
    for (size_t b = 0; b < numBuf; b++) {
        size_t cumulative = 0;
        for (size_t c = 0; c < numChunks; c++) {
            cumulative += compChunksSize[b][c];
            sizePtr[b * numChunks + c] = cumulative;
        }
    }
    offset += cumulativeChunksSizeLen;

    // Calculate number of threads per buffer
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int threads_per_buffer = 4; // Default value
    if (numChunks > 1000) {
        threads_per_buffer = num_cores / numBuf;
        if (threads_per_buffer < 2) threads_per_buffer = 2;
        if (threads_per_buffer > 8) threads_per_buffer = 8;
    }

    const size_t total_threads = numBuf * threads_per_buffer;
    pthread_t *threads = malloc(total_threads * sizeof(pthread_t));
    struct CompressedDataCopyArgs *thread_args = malloc(total_threads * sizeof(struct CompressedDataCopyArgs));
    size_t *threadOffsets = calloc(total_threads, sizeof(size_t));

    if (!threads || !thread_args || !threadOffsets) {
        free(resultBuf);
        free(threads);
        free(thread_args);
        free(threadOffsets);
        return NULL;
    }

    // Launch threads for compressed data copy
    size_t thread_idx = 0;
    for (size_t b = 0; b < numBuf; b++) {
        size_t chunks_per_thread = (numChunks + threads_per_buffer - 1) / threads_per_buffer;
        size_t base_offset = offset;
        
        // Calculate base offset for this buffer
        for (size_t prev_b = 0; prev_b < b; prev_b++) {
            base_offset += totalCompressedSize[prev_b];
        }

        for (int t = 0; t < threads_per_buffer; t++) {
            thread_args[thread_idx] = (struct CompressedDataCopyArgs){
                .b = b,
                .thread_id = thread_idx,
                .start_chunk = t * chunks_per_thread,
                .end_chunk = (t + 1) * chunks_per_thread,
                .numChunks = numChunks,
                .compressedData = compressedData,
                .compChunksSize = compChunksSize,
                .resultBuf = resultBuf,
                .baseOffset = base_offset,
                .threadOffsets = threadOffsets
            };

            if (pthread_create(&threads[thread_idx], NULL, copy_compressed_data, 
                             &thread_args[thread_idx]) != 0) {
                for (size_t i = 0; i < thread_idx; i++) {
                    pthread_join(threads[i], NULL);
                }
                free(threads);
                free(thread_args);
                free(threadOffsets);
                free(resultBuf);
                return NULL;
            }
            thread_idx++;
        }
    }

    // Wait for all threads
    for (size_t i = 0; i < total_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Cleanup
    free(threads);
    free(thread_args);
    free(threadOffsets);
    
    return resultBuf;
}



////////////////////////////////////////////////////////////
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


// Compression //

typedef struct {
    Py_buffer *data;
    size_t numChunks;           
    size_t origChunkSize;
    int numBuf;
    int bits_mode;
    int bytes_mode;
    int is_redata;
    int threads;
    uint8_t ***buffers;
    size_t **unCompChunksSize;
    uint8_t ***compressedData;
    uint32_t **compChunksSize;
    uint8_t **compChunksType;
    int *isThCheck;
    int checkCompTh;
    double compThreshold;
    pthread_mutex_t *next_chunk_mutex;
    size_t *next_chunk;
} CompressionThreadData;

static void* compression_worker(void* arg) {
    CompressionThreadData* thread_data = (CompressionThreadData*)arg;
    size_t current_chunk;
    
    while (1) {
        // Get next chunk to process
//	printf ("start\n");
        pthread_mutex_lock(thread_data->next_chunk_mutex);
        current_chunk = (*thread_data->next_chunk)++;
        pthread_mutex_unlock(thread_data->next_chunk_mutex);

        // Exit if all chunks have been processed
//	printf ("current_chunk %zu\n", current_chunk);
//	printf ("thread_data->numChunks %zu\n",thread_data->numChunks);
        if (current_chunk >= thread_data->numChunks) {
            break;
        }

        // Calculate offset and chunk size
        size_t offset = current_chunk * thread_data->origChunkSize;
//	printf ("offset %zu\n", offset);
        size_t curOrigChunkSize = (current_chunk == thread_data->numChunks - 1) 
            ? (thread_data->data->len - offset)  // Last chunk
            : thread_data->origChunkSize;        // Regular chunk
        
//        printf ("curOrigChunkSize %zu\n", curOrigChunkSize);	

        // Byte Grouping + Byte Ordering
//	printf("thread_data->buffers: %p\n", (void*)thread_data->buffers);
// printf("current_chunk: %zu\n", current_chunk);
//uint8_t **chunk_ptr = thread_data->buffers[current_chunk];
//printf("chunk_ptr: %p\n", (void*)chunk_ptr);
        if (thread_data->numBuf == 2) {
            if (split_bytearray_dtype16(thread_data->data->buf + offset, curOrigChunkSize,
                                      thread_data->buffers[current_chunk], 
                                      thread_data->unCompChunksSize[current_chunk],
                                      thread_data->bits_mode, thread_data->bytes_mode, 
                                      thread_data->is_redata, thread_data->threads) != 0) {
                pthread_exit((void*)-1);
            }
        } else {  // numBuf == 4
            if (split_bytearray_dtype32(thread_data->data->buf + offset, curOrigChunkSize,
                                      thread_data->buffers[current_chunk], 
                                      thread_data->unCompChunksSize[current_chunk],
                                      thread_data->bits_mode, thread_data->bytes_mode, 
                                      thread_data->is_redata, thread_data->threads) != 0) {
                pthread_exit((void*)-1);
            }
        }

       for (int b = 0; b < thread_data->numBuf; b++) {
        // Allocate memory for compressed data
//             printf ("help0000 \n");
        thread_data->compressedData[b][current_chunk] = malloc(thread_data->origChunkSize);
        if (!thread_data->compressedData[b][current_chunk]) {
            pthread_exit((void*)-1);
        }
    
        if (thread_data->buffers[current_chunk][b] != NULL) {
            // Always try to compress initially
            size_t uncompSize = thread_data->unCompChunksSize[current_chunk][b];
//            printf ("help1000 \n");

//            printf ("b %zu current_chunk %zu\n", b, current_chunk);
//            printf ("thread_data->compressedData[b][current_chunk] %zu\n", thread_data->compressedData[b][current_chunk]);
//            printf (" thread_data->origChunkSize %zu\n", thread_data->origChunkSize);
//            printf (" thread_data->buffers[current_chunk][b] %zu\n", thread_data->buffers[current_chunk][b]);

            thread_data->compChunksSize[b][current_chunk] =
                HUF_compress(thread_data->compressedData[b][current_chunk],
                           thread_data->origChunkSize,
                           thread_data->buffers[current_chunk][b],
                           uncompSize);
    
            if (thread_data->compChunksSize[b][current_chunk] != 0 &&
                thread_data->compChunksSize[b][current_chunk] <
                (uncompSize * thread_data->compThreshold)) {  // Fixed parentheses
                thread_data->compChunksType[b][current_chunk] = 1;  // Compress with Huffman
            } else {  // the buffer was not compressed
                free(thread_data->compressedData[b][current_chunk]);
                thread_data->compChunksSize[b][current_chunk] = uncompSize;
                thread_data->compChunksType[b][current_chunk] = 0;  // not compressed
                thread_data->compressedData[b][current_chunk] =
                    thread_data->buffers[current_chunk][b];
            }
        }
      }
    }
    pthread_exit(NULL);
}

PyObject *py_split_dtype(PyObject *self, PyObject *args) {
  Py_buffer header, data;
  int numBuf, bits_mode, bytes_mode, is_redata, checkThAfterPercent, threads;
  size_t origChunkSize;
  float compThreshold;
  // u_int8_t isPrint = 0;

  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);
  if (!PyArg_ParseTuple(args, "y*y*iiiinfii", &header, &data, &numBuf,
                        &bits_mode, &bytes_mode, &is_redata, &origChunkSize,
                        &compThreshold, &checkThAfterPercent, &threads)) {
    return NULL;
  } 

  // Byte Group per chunk, Compress per bufChunk
  size_t numChunks = (data.len + origChunkSize - 1) / origChunkSize;


  uint8_t ***buffers = malloc(numChunks * sizeof(uint8_t **)); //[numChunks][numBuf]
  size_t **unCompChunksSize = malloc(numChunks * sizeof(size_t *));//[numChunks][numBuf]
  for(int c = 0; c < numChunks; c++) {
    buffers[c] = malloc(numBuf * sizeof(uint8_t *));
    unCompChunksSize[c] = malloc(numBuf * sizeof(size_t));
  }

  u_int8_t ***compressedData = malloc(numBuf * sizeof(uint8_t**)); // [numBuf][numChunks]
  u_int8_t **compChunksType = malloc(numBuf * sizeof(uint8_t*));// [numBuf][numChunks]
  uint32_t **compChunksSize = malloc(numBuf * sizeof(uint32_t*));// [numBuf][numChunks]
  for(int b = 0; b < numBuf; b++) {
    compressedData[b] = malloc( numChunks * sizeof(uint8_t*));	  
    compChunksType[b] = malloc( numChunks * sizeof(uint8_t));
    compChunksSize[b] = malloc( numChunks * sizeof(uint32_t));
  }
   
  size_t totalCompressedSize[numBuf];
  size_t totalUnCompressedSize[numBuf];
  size_t cumulativeChunksSize[numBuf][numChunks];
  u_int8_t isThCheck[numBuf];
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
  }

  struct timeval startTimeReal, endTimeReal;
  gettimeofday(&startTimeReal, NULL);
  pthread_t *thread_handles = NULL;
  CompressionThreadData *thread_data = NULL;
  pthread_mutex_t compression_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t next_chunk_mutex = PTHREAD_MUTEX_INITIALIZER;
  size_t next_chunk = 0;

  thread_handles = malloc(threads * sizeof(pthread_t));
  thread_data = malloc(threads * sizeof(CompressionThreadData));
  if (!thread_handles || !thread_data) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate thread resources");
    goto cleanup_threads;
  }

  // Create threads
  for (int i = 0; i < threads; i++) {
    thread_data[i] = (CompressionThreadData){
        .data = &data,
	.numChunks = numChunks,
        .origChunkSize = origChunkSize,
        .numBuf = numBuf,
        .bits_mode = bits_mode,
        .bytes_mode = bytes_mode,
        .is_redata = is_redata,
        .threads = threads,
        .buffers = buffers,
        .unCompChunksSize = unCompChunksSize,
        .compressedData = compressedData,
        .compChunksSize = compChunksSize,
        .compChunksType = compChunksType,
        .isThCheck = isThCheck,
        .checkCompTh = checkCompTh,
        .compThreshold = compThreshold,
        .next_chunk_mutex = &next_chunk_mutex,
        .next_chunk = &next_chunk
    };
    
    if (pthread_create(&thread_handles[i], NULL, compression_worker, &thread_data[i]) != 0) {
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
  goto continue_processing;

  cleanup_threads:
    if (thread_handles) free(thread_handles);
    if (thread_data) free(thread_data);
    pthread_mutex_destroy(&next_chunk_mutex);
    return NULL;

  ////////////// The end of multi Threading part 1
  /////////////////////////////////
continue_processing: 

printf("compChunksSize[%zu][%zu] %zu\n ", 0, 0, compChunksSize[0][0]);
printf("compChunksSize[%zu][%zu] %zu\n ", 0, 1, compChunksSize[0][1]);

for (int b = 0; b < numBuf; b++) {
    size_t totalCompressed = 0;
    size_t totalUncompressed = 0;
    
    for (size_t chunk = 0; chunk < numChunks; chunk++) {
        totalCompressed += compChunksSize[b][chunk];
        totalUncompressed += unCompChunksSize[chunk][b];
        cumulativeChunksSize[b][chunk] = totalCompressed;
        
        // Check if compression is beneficial after checkCompTh chunks
        if (chunk == checkCompTh && 
            totalCompressed * 1.0 > totalUncompressed * compThreshold) {
//            noNeedToCompress[b] = 1;
        }
    }
    totalCompressedSize[b] = totalCompressed;
    totalUnCompressedSize[b] = totalUncompressed;
}
  gettimeofday(&endTimeReal, NULL);
  double compressMl1TimeReal = (endTimeReal.tv_sec - startTimeReal.tv_sec) + 
                            (endTimeReal.tv_usec - startTimeReal.tv_usec) / 1e6;
    printf("compress ML1: %f seconds\n", compressMl1TimeReal);

  PyObject *py_result;
  u_int8_t *resultBuf;
  size_t resBufSize;

  printf("start prepare: %f seconds\n", compressMl1TimeReal);
   printf ("compChunksSize[0][0] %u\n", compChunksSize[0][0]);
  resultBuf = prepare_split_results(
      header.len, numBuf, numChunks, header.buf, compressedData, compChunksSize,
      compChunksType, cumulativeChunksSize, totalCompressedSize, &resBufSize);
  gettimeofday(&endTimeReal, NULL);
  double compressPrepareTimeReal = (endTimeReal.tv_sec - startTimeReal.tv_sec) + 
                            (endTimeReal.tv_usec - startTimeReal.tv_usec) / 1e6;
   printf("compress Prepare: %f seconds\n", compressPrepareTimeReal);

  if (resultBuf == NULL) {
    // Free all Mallocs
    // print Error
    return NULL;
  }

  gettimeofday(&startTimeReal, NULL);

  Py_buffer view; // create buffer to avoid copy
  PyBuffer_FillInfo(&view, NULL, resultBuf, resBufSize, 0, PyBUF_WRITABLE);
  py_result = PyMemoryView_FromBuffer(&view);

  // Freeing compressedData array
//   for (uint32_t c = 0; c < numChunks; c++) {
//     for (int b = 0; b < numBuf; b++) {
//       if (buffers[c][b] != NULL) {
//         free(buffers[c][b]);
//       }
//     }
//   }

  gettimeofday(&endTimeReal, NULL);
  double freeTimeReal = (endTimeReal.tv_sec - startTimeReal.tv_sec) + 
                            (endTimeReal.tv_usec - startTimeReal.tv_usec) / 1e6;
  printf("compress free: %f seconds\n", freeTimeReal);



  gettimeofday(&endTime, NULL);
  double compressAll = (endTime.tv_sec - startTime.tv_sec) + 
                            (endTime.tv_usec - startTime.tv_usec) / 1e6;
  printf("compress All: %f seconds\n", compressAll);

  return py_result;
}


////////////////////////////////////////////////////////////////////////////
//////////////////////   Decompression ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////




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
	int freeDeCompressedDataPtr[data->numBuf];
        for (int b = 0; b < data->numBuf; b++) {
            // Access 2D array [b][current_chunk]
            if (data->compChunksType[b * data->chunk_id + current_chunk] == 0) {
	        freeDeCompressedDataPtr[b] = 0;
                data->deCompressedDataPtr[b][current_chunk] = 
                    data->ptrCompressData[b] + 
                    data->compCumulativeChunksPos[b * (data->chunk_id + 1) + current_chunk];
            } else if (data->compChunksType[b * data->chunk_id + current_chunk] == 1) {
                // Get decompLen[current_chunk][b]
                size_t decomp_length = data->decompLen[current_chunk * data->numBuf + b];

  	       data->deCompressedDataPtr[b][current_chunk] = malloc(decomp_length);
	       freeDeCompressedDataPtr[b] = 1;
               if (!data->deCompressedDataPtr[b][current_chunk]) {
                   pthread_exit((void*)-1);
              }
                
	       size_t decompressedSize = HUF_decompress(
			       data->deCompressedDataPtr[b][current_chunk],
			       decomp_length,
			       (void*)(data->ptrCompressData[b] + 
				       data->compCumulativeChunksPos[b * (data->chunk_id + 1) + current_chunk]),
			       data->compChunksLen[b * data->chunk_id + current_chunk]);
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
        for (int b = 0; b < data->numBuf; b++) {
	  if (freeDeCompressedDataPtr[b] == 1){
	    free(data->deCompressedDataPtr[b][current_chunk]);	  
	  }	  
	}
    }
    
    pthread_exit(NULL);
}


// Python callable function to combine four buffers into a single bytearray
PyObject *py_combine_dtype(PyObject *self, PyObject *args) {
  Py_buffer data;
   
  //clock_t sTime, eTime;
  //sTime = clock();
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
	  printf ("compChunksType[0][0] %zu\n", compChunksType[0][0]);
	  printf ("compChunksType[1][0] %zu\n", compChunksType[1][0]);
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
  //eTime = clock();
  //double metadataTime = (double)(eTime - sTime) / CLOCKS_PER_SEC;
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
  PyObject *py_result;  
  continue_processing:
//   endTime = clock();
//   double decompressTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
   gettimeofday(&endTimeReal, NULL);
   double decompressTimeReal = (endTimeReal.tv_sec - startTimeReal.tv_sec) + 
                            (endTimeReal.tv_usec - startTimeReal.tv_usec) / 1e6;
//   printf ("thread decompressTime %f\n", decompressTime);
   printf("Real thread time: %f seconds\n", decompressTimeReal);

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
