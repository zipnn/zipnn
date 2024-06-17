#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <time.h>
#include "split_dtype_functions.h"

///////////////////////////////////
/// Split Helper Functions ///////
//////////////////////////////////

// Reordering function for float bits
static uint32_t reorder_float_bits(float number) {
    union {
        float f;
        uint32_t u;
    } value = { .f = number };

    uint32_t sign = (value.u >> 8) & 0x800080;
    uint32_t exponent = (value.u << 1) & 0xFF00FF00;
    uint32_t mantissa = (value.u) & 0x7F007F;
    return exponent | sign | mantissa;  
}

// Helper function to reorder all floats in a bytearray
static void reorder_all_floats(char *src, Py_ssize_t len) {
    uint32_t *uint_array = (uint32_t *)src;
    Py_ssize_t num_floats = len / sizeof(uint32_t);
    for (Py_ssize_t i = 0; i < num_floats; i++) {
        uint_array[i] = reorder_float_bits(*(float *)&uint_array[i]);
    }
}

// Helper function to split a bytearray into four buffers
static int split_bytearray(char *src, Py_ssize_t len, char **buf1, char **buf2, int bits_mode, int bytes_mode, int is_review,  int threads) {
	
    if (bits_mode == 1) { // reoreder exponent 
        reorder_all_floats(src, len);
    }

    Py_ssize_t half_len = len / 2;
    switch(bytes_mode){ 
	    case 6:     // 2b0110 - Byte Group to two different groups
	        *buf1 = PyMem_Malloc(half_len);
        	*buf2 = PyMem_Malloc(half_len);

	        if (*buf1 == NULL || *buf2 == NULL) {
		        PyMem_Free(*buf1);
        		PyMem_Free(*buf2);
		        return -1;
        	}

		char *dst1 = *buf1;
		char *dst2 = *buf2;
	
		for (Py_ssize_t i = 0; i < len; i += 2) {
		        *dst1++ = src[i];
	        	*dst2++ = src[i + 1];
	    	}
	        break;
	    
	    case 8:  // 4b1000 - Truncate MSByte 
		// We are refering to the MSBbyte as little endian, thus we omit buf2 
	    case 1:  // 4b1000 - Truncate LSByte 
		// We are refering to the LSByte  as a little endian, thus we omit buf1 
	        *buf1 = PyMem_Malloc(half_len);
	        *buf2 = NULL;
	
	        if (*buf1 == NULL) {
		        PyMem_Free(*buf1);
		        return -1;
	        }
	
		dst1 = *buf1;
	        
		if (bytes_mode == 1) {
			for (Py_ssize_t i = 0; i < len; i += 2) {
				*dst1++ = src[i];
			}
		}
		else{
			for (Py_ssize_t i = 0; i < len; i += 2) {
			        *dst1++ = src[i+1];
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
    } value = { .f = number };

    uint32_t sign = (value.u << 8) & 0x80008000;
    uint32_t exponent = (value.u >> 1) & 0x7F807F80;
    uint32_t mantissa = (value.u) & 0x7F007F;
    return sign | exponent | mantissa;  
}


// Helper function to reorder all floats in a bytearray
static void revert_all_floats(char *src, Py_ssize_t len) {
    uint32_t *uint_array = (uint32_t *)src;
    Py_ssize_t num_floats = len / sizeof(uint32_t);
    for (Py_ssize_t i = 0; i < num_floats; i++) {
        uint_array[i] = revert_float_bits(*(float *)&uint_array[i]);
    }
}


// Helper function to combine four buffers into a single bytearray
static char* combine_buffers(char *buf1, char *buf2, Py_ssize_t half_len ,int bytes_mode, int threads) {
    Py_ssize_t total_len = half_len * 2;
    char *result = NULL;  // Declare result at the beginning of the function
    char *dst;
    result = PyMem_Malloc(total_len);
    dst = result;
    if (result == NULL) {
	    return NULL;
    }
		    
    switch(bytes_mode){ 
	    case 6:     // 2b0110 - Byte Group to two different groups

		    if (result == NULL) {
			    return NULL;
		    }

		    for (Py_ssize_t i = 0; i < half_len; i++) {
			    *dst++ = buf1[i];
			    *dst++ = buf2[i];
			    }
	        break;
	    
	    case 8:  // 4b1000 - Truncate MSByte
		    // We are refering to the MSByte as a little endian, thus we omit buf2 
	    case 1:  // 4b1000 - Truncate LSByte 
		    // We are refering to the LSByte as a little endian, thus we omit buf1 

	            if (bytes_mode == 8){	    
			    for (Py_ssize_t i = 0; i < half_len; i++) {
				    *dst++ = 0;
				    *dst++ = buf1[i];
			    }
		    }
		    else{
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
//     [we are refering to the bytes order as first 2bits refer to the MSByte and the second two bits to the LSByte]
//     2b [MSB Byte],2b[LSB Byte]
//     0 - truncate this byte
//     1 or 2 - a group of bytes
//     4b0110 [6] - bytegroup to two groups
//     4b0001 [1] - truncate the MSByte       
//     4b1000 [8] - truncate the LSByte     
// is_review:
//     Even if you have the Byte mode, you can change it if needed.
//     0 - No review, take the bit_mode and byte_mode
//     1 - the finction can change the Bytes_mode       
      
PyObject* py_split_dtype16(PyObject *self, PyObject *args) {
    Py_buffer view;
    int bits_mode, bytes_mode, is_review, threads;

    if (!PyArg_ParseTuple(args, "y*iiii", &view, &bits_mode, &bytes_mode, &is_review, &threads)) {
        return NULL;
    }

    char *buf1 = NULL, *buf2 = NULL;
    if (split_bytearray(view.buf, view.len, &buf1, &buf2, bits_mode, bytes_mode, is_review,  threads) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }
   
    PyObject *result; 
    if (buf2 != NULL) {
    	result = Py_BuildValue("y#y#", buf1, view.len / 2, buf2, view.len / 2);
    } 
    else {
    	result = Py_BuildValue("y#O", buf1, view.len / 2, Py_None);
    }

    PyMem_Free(buf1);
    PyMem_Free(buf2);
    PyBuffer_Release(&view);

    return result;
}



// Python callable function to combine four buffers into a single bytearray
PyObject* py_combine_dtype16(PyObject *self, PyObject *args) {
    Py_buffer view1, view2;
    int bits_mode, bytes_mode, threads;

    if (!PyArg_ParseTuple(args, "y*y*iii", &view1, &view2, &bits_mode, &bytes_mode, &threads)) {
        return NULL;
    }

    char *result = combine_buffers((char *)view1.buf, (char *)view2.buf, view1.len, bytes_mode, threads);
    if (result == NULL) {
        PyBuffer_Release(&view1);
        PyBuffer_Release(&view2);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
        return NULL;
    }

    // Revert the reordering of all floats if needed
    if (bits_mode == 1) {
        revert_all_floats(result, view1.len * 2);
    }

    PyObject *py_result = PyByteArray_FromStringAndSize(result, view1.len * 2);
    PyMem_Free(result);
    PyBuffer_Release(&view1);
    PyBuffer_Release(&view2);

    return py_result;
}
