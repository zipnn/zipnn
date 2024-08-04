#ifndef HUF_API_H
#define HUF_API_H

#include <stddef.h>
#include <stdlib.h>

size_t hufCompressData(const u_int8_t *data, size_t size, int maxCompressedSize,
                       u_int8_t *compressedData, size_t *compressedChunksSize,
                       size_t chunkSize, float compThreshold,
                       size_t checkThreshold);
size_t hufDecompressData(const u_int8_t *compressedData,
                         size_t *compressedChunksSize, size_t original_size,
                         size_t numChunks, u_int8_t *decompressedData,
                         size_t chunkSize);

#endif // HUF_API_H
