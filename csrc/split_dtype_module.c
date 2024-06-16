#include <Python.h>
#include "split_dtype_functions.h"

// Declare functions from other source files
extern PyObject* py_split_dtype16(PyObject*, PyObject*);
extern PyObject* py_combine_dtype16(PyObject*, PyObject*);
extern PyObject* py_split_dtype32(PyObject*, PyObject*);
extern PyObject* py_combine_dtype32(PyObject*, PyObject*);

static PyMethodDef SplitMethods[] = {
    {"split_dtype16", py_split_dtype16, METH_VARARGS, "Split a bytearray into four buffers using dtype16"},
    {"combine_dtype16", py_combine_dtype16, METH_VARARGS, "Combine four buffers into a single bytearray using dtype16"},
    {"split_dtype32", py_split_dtype32, METH_VARARGS, "Split a bytearray into four buffers using dtype32"},
    {"combine_dtype32", py_combine_dtype32, METH_VARARGS, "Combine four buffers into a single bytearray using dtype32"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef splitmodule = {
    PyModuleDef_HEAD_INIT,
    "split_dtype",  // Correct the module name here
    NULL,  // Optional module docstring
    -1,    // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    SplitMethods
};

PyMODINIT_FUNC PyInit_split_dtype(void) {  // Correct the function name here
    return PyModule_Create(&splitmodule);
}

