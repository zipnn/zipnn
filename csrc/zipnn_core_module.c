#include "zipnn_core_functions.h"
#include <Python.h>

// Declare functions from other source files
extern PyObject *py_zipnn_core(PyObject *, PyObject *);
extern PyObject *py_combine_dtype(PyObject *, PyObject *);

// Method definitions
static PyMethodDef SplitMethods[] = {
    {"zipnn_core", py_zipnn_core, METH_VARARGS,
     "Split a bytearray into four buffers using dtype16"},
    {"combine_dtype", py_combine_dtype, METH_VARARGS,
     "Combine four buffers into a single bytearray using dtype16"},
};

// Module definition
static struct PyModuleDef splitmodule = {PyModuleDef_HEAD_INIT, "zipnn_core",
                                         NULL, -1, SplitMethods};

// Module initialization function
PyMODINIT_FUNC PyInit_zipnn_core(void) {
  return PyModule_Create(&splitmodule);
}
