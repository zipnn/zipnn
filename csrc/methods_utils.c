#include <stdint.h>
#include <time.h>
#include <Python.h>
#include "methods_enums.h"
#include "methods_utils.h"

////////////////////////////////////////////////////////////////////
//// Helper function for choosing methods according to zero bytes //
////////////////////////////////////////////////////////////////////

const char* getEnumName(MethodsEnums method) {
    switch(method) {
        case ORIGINAL:
            return "ORIGINAL";
        case HUFFMAN:
            return "HUFFMAN";
        case ZSTD:
            return "ZSTD";
        case FSE:
            return "FSE";
        case TRUNCATE:
            return "TRUNCATE";
        case AUTO:
            return "AUTO";
        default:
            return "UNKNOWN";
    }
}


int calc_chunk_methods_dtype32(size_t *zeroCount, size_t *maxSeqZeros, int *chunk_methods, const Py_ssize_t num_buf_len, const int num_buf) {
  // LOGIC:
  // If all bytes are zeros, set the method to TRUNCATE.
  //  If the number of zeros is above 90% or if the sequence of zeros is above 10%, set the method to ZSTD.
  // Otherwise, set the method to HUFFMAN.
  //
  float zeroCountForZSTD = 0.92;
  float zeroSeqCountForZSTD = 0.03;
				 //
  for (int b=0; b < num_buf; b++) {
//    printf ("zeroCount[%d] %zu, len %zu , zeroCount[b]/len %f\n", b, zeroCount[b], num_buf_len, zeroCount[b]*1.0/num_buf_len);  	  
    if (zeroCount[b] == num_buf_len) {
    // all zeros Truncate 
      chunk_methods[b] = TRUNCATE;
    }
    else {
	  float zeroCountPrecent;
	  float zeroSeqPrecent;
	  if (num_buf == 4) {
	    zeroCountPrecent =  zeroCount[b] *1.0 / num_buf_len; 
	    zeroSeqPrecent = maxSeqZeros[b] *1.0 / num_buf_len; 
	  }  
	  else { // num_buf ==2
	    zeroCountPrecent = (zeroCount[b] + zeroCount[b+2] *1.0) / num_buf_len; 
	    zeroSeqPrecent = (maxSeqZeros[b] *1.0) / num_buf_len; 
	  }
//          printf ("zeroCountPrecent %f\n", zeroCountPrecent);
//          printf ("zeroSeqPrecent %f\n", zeroSeqPrecent);
	  if ((zeroCountPrecent > zeroCountForZSTD) || (zeroSeqPrecent > zeroSeqCountForZSTD)) {
            chunk_methods[b] = ZSTD;
	  }
	  else {
            chunk_methods[b] = HUFFMAN;
	  }
    }
  }
}
 

