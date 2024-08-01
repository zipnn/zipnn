#include "huf_wrapper.h"
#include "huf.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

size_t hufCompressData(const uint8_t *data, size_t size, int maxCompressedSize,
                       uint8_t *compressedData, size_t *compressedChunksSize,
                       size_t chunkSize, float compThreshold,
                       size_t checkThreshold) {
  size_t curChunk = 0;
  size_t totalCompressedSize = 0;
  size_t RemianCapacity = maxCompressedSize;
  size_t checkCompTh =
      size / checkThreshold;  // check after checkTh% of the size,
			      //  if not compressed -> stop compressing
  uint8_t isCheck = 0;

  for (size_t offset = 0; offset < size; offset += chunkSize) {
    size_t curChunkSize =
        (size - offset > chunkSize) ? chunkSize : (size - offset);

    // Compress
    size_t compressedSize =
        HUF_compress(compressedData + totalCompressedSize, RemianCapacity,
                     data + offset, curChunkSize);

    if (compressedSize == 0) {
      memcpy(compressedData + totalCompressedSize, data + offset, curChunkSize);
      compressedSize = curChunkSize;
    }

    totalCompressedSize += compressedSize;

    if (isCheck == 0 & offset > checkCompTh) {
      isCheck = 1;
      if (compThreshold * offset < totalCompressedSize * 1.0) {
        return 0;
      }
    }

    if (HUF_isError(compressedSize)) {
      printf("Compression failed: %s\n", HUF_getErrorName(compressedSize));
      return -1;
    }

    RemianCapacity -= compressedSize;
    if (maxCompressedSize < 0) {
      printf("Not enough space for compression : %s\n",
             HUF_getErrorName(compressedSize));
      return -1;
    }
    compressedChunksSize[curChunk++] = totalCompressedSize;
  }
  return totalCompressedSize;
}

size_t hufDecompressData(const uint8_t *compressedData,
                         size_t *compressedChunksSize, size_t numChunks,
                         size_t original_size, uint8_t *decompressedData,
                         size_t chunkSize) {
  size_t totalDecompressedSize = 0;
  size_t compressedOffset = 0;

  for (size_t i = 0; i < numChunks; i++) {
    size_t curChunkSize =
        (i > 0) ? compressedChunksSize[i] - compressedChunksSize[i - 1]
                : compressedChunksSize[0];
    size_t remainingOriginalSize = original_size - totalDecompressedSize;
    size_t expectedChunkSize =
        (remainingOriginalSize > chunkSize) ? chunkSize : remainingOriginalSize;

    // Decompress each chunk
    size_t decompressedSize = HUF_decompress(
        decompressedData + totalDecompressedSize, expectedChunkSize,
        compressedData + compressedOffset, curChunkSize);

    if (HUF_isError(decompressedSize)) {
      printf("Decompression failed for chunk %zu: %s\n", i,
             HUF_getErrorName(decompressedSize));
      return 0;
    }

    totalDecompressedSize += decompressedSize;
    compressedOffset += curChunkSize;
  }

  return totalDecompressedSize;
}
