#define PY_SSIZE_T_CLEAN
#include "split_dtype_functions.h"
#include <Python.h>
#include <stdint.h>
#include <time.h>
#include "data_manipulation_dtype32.h" 

/////////////////////////////////////////////////////////////
//////////////// Python callable Functions /////////////////
/////////////////////////////////////////////////////////////

// Python callable function to split a bytearray into four buffers
// bits_mode:
//     0 - no ordering of the bits
//     1 - reorder of the exponent (eponent, sign_bit, mantissa)
// bytes_mode:
//     [we are refering to the bytes order as first 2bits refer to the MSByte
//     and the second two bits to the LSByte] 1b [MSByte],2b[MID-HIGH Byte],
//     2b[MID-LOW Byte], 3b[LSByte] 0 - truncate this byte 1-4 - different
//     groups 8b1_10_11_100 [decimal 220] - bytegroup to four groups [1,2,3,4]
//     8b0_01_01_001 [decimal 41] - truncate the MSB [0,1,1,1]
//     8b0_00_01_001 [decimal 9] - truncate the MSB+MID_HIGH [0,0,1,1]
//     8b0_00_00_001 [decimal 1] - truncate the MSB+MID_HIGH+MID_LOW [0,0,0,1]
//
//     8b0_01_10_010 [decimal 50] - Truncate MSB, Group 1 MID-HIGH, group 2
//     MID_LOW+LSB [0,1,2,2] 8b0_00_10_010 [decimal 18] - Truncate MSB,
//     MID-HIGH, Group 1 MID-LOW, group 2 LSB [0,0,1,2] 8b0_01_01_011 [decimal
//     43] - Truncate MSB, Group 1 MID-LOW, Group 2 MID_LOW, GROUP 3 LSB
//     [0,1,2,3]
//
//     8b1_10_10_010 [decimal 210] - Group 1 MSB, Group 2 MID-LOW, MID_LOW, LSB
//     [1,2,2,2] 8b1_10_00_000 [decimal 192] - Group 1 MSB, Group 2 MID-LOW,
//     Truncate MID_LOW, LSB  [1,2,0,0] 8b1_10_10_000 [decimal 208] - Group 1
//     MSB, Group 2 MID-LOW, MID_LOW, Truncate LOW  [1,2,2,0]
//
// is_review:
//     Even if you have the Byte mode, you can change it if needed.
//     0 - No review, take the bit_mode and bytes_mode
//     1 - the finction can change the Bytes_mode

PyObject *py_split_dtype32(PyObject *self, PyObject *args) {
  Py_buffer view;
  int bits_mode, bytes_mode, is_review, threads;

  if (!PyArg_ParseTuple(args, "y*iiii", &view, &bits_mode, &bytes_mode,
                        &is_review, &threads)) {
    return NULL;
  }

  u_int8_t *buf1 = NULL, *buf2 = NULL, *buf3 = NULL, *buf4 = NULL;
  Py_ssize_t buf1_len = 0, buf2_len = 0, buf3_len = 0, buf4_len = 0;
  if (split_dtype32(view.buf, view.len, &buf1, &buf2, &buf3, &buf4, &buf1_len,
                    &buf2_len, &buf3_len, &buf4_len, bits_mode, bytes_mode,
                    is_review, threads) != 0) {
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    return NULL;
  }

  PyObject *result = Py_BuildValue("y#y#y#y#", buf1, buf1_len, buf2, buf2_len,
                                   buf3, buf3_len, buf4, buf4_len);
  PyMem_Free(buf1);
  PyMem_Free(buf2);
  PyMem_Free(buf3);
  PyMem_Free(buf4);
  PyBuffer_Release(&view);

  return result;
}

// Python callable function to combine four buffers into a single bytearray
PyObject *py_combine_dtype32(PyObject *self, PyObject *args) {
  Py_buffer view1, view2, view3, view4;
  int bits_mode, bytes_mode, threads;

  if (!PyArg_ParseTuple(args, "y*y*y*y*iii", &view1, &view2, &view3, &view4,
                        &bits_mode, &bytes_mode, &threads)) {
    return NULL;
  }

  Py_ssize_t total_len = 0;
  u_int8_t *result =
      combine_dtype32(&total_len, (u_int8_t *)view1.buf, (u_int8_t *)view2.buf,
                      (u_int8_t *)view3.buf, (u_int8_t *)view4.buf, view1.len,
                      view2.len, view3.len, view4.len, bytes_mode, threads);

  if (result == NULL) {
    PyBuffer_Release(&view1);
    PyBuffer_Release(&view2);
    PyBuffer_Release(&view3);
    PyBuffer_Release(&view4);
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    return NULL;
  }

  // Revert the reordering of all floats if needed
  if (bits_mode == 1) {
    revert_all_floats(result, total_len);
  }

  PyObject *py_result = PyByteArray_FromStringAndSize(result, total_len);
  PyMem_Free(result);
  PyBuffer_Release(&view1);
  PyBuffer_Release(&view2);
  PyBuffer_Release(&view3);
  PyBuffer_Release(&view4);

  return py_result;
}
