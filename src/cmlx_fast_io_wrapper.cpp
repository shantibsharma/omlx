#include <Python.h>

/*
 * This module acts as a thin wrapper around the C functions exported
 * from the compiled C++ source files.
 *
 * When compiled via setuptools, the functions from cache_core.cpp,
 * scheduler_core.cpp, etc., will be part of this module's namespace.
 *
 * We define '_lib' here to satisfy the existing `cmlx/c_bindings.py`
 * logic which expects `from . import cmlx_fast_io as _lib_module`
 * followed by `_lib = _lib_module._lib`.
 */

static PyObject* get_lib(PyObject* self, PyObject* args) {
    // Return self (the module object) to satisfy references to '_lib'
    Py_INCREF(self);
    return self;
}

static PyMethodDef ModuleMethods[] = {
    {"_lib", (PyCFunction)get_lib, METH_NOARGS, "Returns the module itself to act as the ctypes library object."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cmlx_fast_io_module = {
    PyModuleDef_HEAD_INIT,
    "cmlx_fast_io",
    "Native C++ acceleration for cMLX",
    -1,
    ModuleMethods
};

PyMODINIT_FUNC PyInit_cmlx_fast_io(void) {
    return PyModule_Create(&cmlx_fast_io_module);
}
