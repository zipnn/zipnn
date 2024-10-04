#include "split_dtype_functions.h"
#include <Python.h>

// Declare functions from other source files
extern PyObject *py_split_dtype(PyObject *, PyObject *);
extern PyObject *py_combine_dtype(PyObject *, PyObject *);

// Method definitions
static PyMethodDef SplitMethods[] = {
    {"split_dtype", py_split_dtype, METH_VARARGS,
     "Split a bytearray into four buffers using dtype16"},
    {"combine_dtype", py_combine_dtype, METH_VARARGS,
     "Combine four buffers into a single bytearray using dtype16"},
};

// Module definition
static struct PyModuleDef splitmodule = {PyModuleDef_HEAD_INIT, "split_dtype",
                                         NULL, -1, SplitMethods};

// Module initialization function
PyMODINIT_FUNC PyInit_split_dtype(void) {
  return PyModule_Create(&splitmodule);
}
