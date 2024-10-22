#ifndef METHODS_UTILS_H
#define METHODS_UTILS_H

#include "methods_enums.h"
////////////////////////////////////////////////////////////////////
//// Helper function for choosing methods according to zero bytes //
////////////////////////////////////////////////////////////////////
int calc_chunk_methods_dtype32(size_t *zeroCount, size_t *maxSeqZeros, int *chunk_methods, const Py_ssize_t num_buf_len, const int num_buf);
#endif // METHODS_UTILS_H
