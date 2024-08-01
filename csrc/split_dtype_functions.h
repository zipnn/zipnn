#ifndef SPLIT_FUNCTIONS_H
#define SPLIT_FUNCTIONS_H

#include <Python.h>

// Declare the functions
PyObject *py_split_dtype16(PyObject *self, PyObject *args);
PyObject *py_combine_dtype16(PyObject *self, PyObject *args);
PyObject *py_split_dtype32(PyObject *self, PyObject *args);
PyObject *py_combine_dtype32(PyObject *self, PyObject *args);

#endif // SPLIT_FUNCTIONS_H
