/*
 * XXX
 */

#define PY_ARRAY_UNIQUE_SYMBOL PYBLAW_CLINEARSOURCE_ARRAY_API

#include <stdio.h>
#include <stdlib.h>

#include <Python.h>
#include <ndarrayobject.h>

/*********************************************************************/

int N, p;
double *B, *s_g, *g;
int n, m;

void
nsource(double *q, double *s)
{
  int i, j;

  for (i=0; i<n; i++) {
    s[i] = B[i*m] * q[0];

    for (j=1; j<p; j++)
      s[i] += B[i*m+j] * q[j];
  }
}

/*********************************************************************/

PyObject *
init_linear_q3_source(PyObject *self, PyObject *args)
{
  PyObject *B_py;

  /*
   * parse options
   */
  if (! PyArg_ParseTuple(args, "O", &B_py))
    return NULL;

  if ((PyArray_FLAGS(B_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "B is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(B_py);

  /*
   * set globals
   */

  B = (double *) PyArray_DATA(B_py);

  n = PyArray_DIM(B_py, 0);
  m = PyArray_DIM(B_py, 1);

  s_g = (double *) malloc(sizeof(double)*n);
  g = (double *) malloc(sizeof(double)*n);

  /*
   * done
   */
  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *
linear_q3_source(PyObject *self, PyObject *args)
{
  long int i;
  PyObject *qq_py, *s_py;
  double *qq, *s;

  int j;

  /*
   * parse options
   */

  if (! PyArg_ParseTuple(args, "OO", &qq_py, &s_py))
    return NULL;

  if ((PyArray_FLAGS(qq_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "qq is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(qq_py);
  qq = (double *) PyArray_DATA(qq_py);

  if ((PyArray_FLAGS(s_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "s is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(s_py);
  s = (double *) PyArray_DATA(s_py);

  /*
   * compute net source
   */

  N = PyArray_DIM(s_py, 0);
  p = PyArray_DIM(s_py, 1);

  /*
   * compute net source (quadrature weights: w1=5/9, w2=8/9, w3=5/9)
   */

  for (i=0; i<N; i++) {
    nsource(qq + i*p*3 + 0*p, s_g);
    for (j=0; j<p; j++)
      g[j] = 5.0/9.0*s_g[j];

    nsource(qq + i*p*3 + 1*p, s_g);
    for (j=0; j<p; j++)
      g[j] = 5.0/9.0*s_g[j];

    nsource(qq + i*p*3 + 2*p, s_g);
    for (j=0; j<p; j++)
      g[j] = 5.0/9.0*s_g[j];

    for (j=0; j<p; j++)
      s[i*p+j] = 0.5 * g[j];
  }

  /*
   * done
   */
  Py_INCREF(Py_None);
  return Py_None;
}

/*********************************************************************/

static PyMethodDef clinearsourcemethods[] = {
    {"init_linear_q3_source", init_linear_q3_source, METH_VARARGS, "XXX"},
    {"linear_q3_source", linear_q3_source, METH_VARARGS, "XXX"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initclinearsource(void)
{
  (void) Py_InitModule("clinearsource", clinearsourcemethods);
  import_array();
}
