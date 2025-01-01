#ifndef SPLIT_FUNCTIONS_H
#define SPLIT_FUNCTIONS_H

#include <Python.h>

// Declare the functions
PyObject *py_zipnn_core(PyObject *self, PyObject *args);
PyObject *py_combine_dtype(PyObject *self, PyObject *args);
PyObject *py_zipnn_core32(PyObject *self, PyObject *args);
PyObject *py_combine_dtype32(PyObject *self, PyObject *args);

#endif // SPLIT_FUNCTIONS_H
