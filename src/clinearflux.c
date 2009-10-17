/*
 * XXX
 */

#define PY_ARRAY_UNIQUE_SYMBOL PYBLAW_CLINEARFLUX_ARRAY_API

#include <stdio.h>
#include <stdlib.h>

#include <Python.h>
#include <ndarrayobject.h>

/*********************************************************************/

int N, p;
double alpha, *f_l, *f_r, *f_m, *f_p, *dx, *A;
int n, m;

void
flux(double *q, double *f)
{
  int i, j;

  for (i=0; i<n; i++) {
    f[i] = A[i*m] * q[0];

    for (j=1; j<p; j++)
      f[i] += A[i*m+j] * q[j];
  }
}

void
nflux_lf(double *q_m, double *q_p, double *f)
{
  int j;

  flux(q_m, f_m);
  flux(q_p, f_p);

  for (j=0; j<p; j++)
    f[j] = 0.5 * (f_m[j] + f_p[j] - alpha * (q_p[j] - q_m[j]) );
}

/*********************************************************************/

PyObject *
init_linear_lf_flux(PyObject *self, PyObject *args)
{
  PyObject *A_py, *dx_py;

  /*
   * parse options
   */
  if (! PyArg_ParseTuple(args, "OdO", &A_py, &alpha, &dx_py))
    return NULL;

  if ((PyArray_FLAGS(A_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "A is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(A_py);

  if ((PyArray_FLAGS(dx_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "dx is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(dx_py);

  /*
   * set globals
   */

  A = (double *) PyArray_DATA(A_py);
  dx = (double *) PyArray_DATA(dx_py);

  n = PyArray_DIM(A_py, 0);
  m = PyArray_DIM(A_py, 1);

  f_l = (double *) malloc(n*sizeof(double));
  f_r = (double *) malloc(n*sizeof(double));
  f_m = (double *) malloc(n*sizeof(double));
  f_p = (double *) malloc(n*sizeof(double));

  /*
   * done
   */
  Py_INCREF(Py_None);
  return Py_None;
}

PyObject *
linear_lf_flux(PyObject *self, PyObject *args)
{
  long int i;
  PyObject *qm_py, *qp_py, *f_py;
  double *qm, *qp, *f;

  int j;

  /*
   * parse options
   */

  if (! PyArg_ParseTuple(args, "OOO", &qm_py, &qp_py, &f_py))
    return NULL;

  if ((PyArray_FLAGS(qm_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "qm is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(qm_py);
  qm = (double *) PyArray_DATA(qm_py);

  if ((PyArray_FLAGS(qp_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "qp is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(qp_py);
  qp = (double *) PyArray_DATA(qp_py);

  if ((PyArray_FLAGS(f_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "f is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(f_py);
  f = (double *) PyArray_DATA(f_py);

  /*
   * compute net flux
   */

  N = PyArray_DIM(f_py, 0);
  p = PyArray_DIM(f_py, 1);

  nflux_lf(qm, qp, f_r);

  for (i=1; i<N; i++) {
    for (j=0; j<p; j++)
      f_l[j] = f_r[j];

    nflux_lf(qm + (i+1)*p, qp + (i+1)*p, f_r);

    for (j=0; j<p; j++)
      f[i*p+j] = - ( f_r[j] - f_l[j] ) / dx[i];
  }

  /*
   * done
   */
  Py_INCREF(Py_None);
  return Py_None;
}

/*********************************************************************/

static PyMethodDef clinearfluxmethods[] = {
    {"init_linear_lf_flux", init_linear_lf_flux, METH_VARARGS, "XXX"},
    {"linear_lf_flux", linear_lf_flux, METH_VARARGS, "XXX"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initclinearflux(void)
{
  (void) Py_InitModule("clinearflux", clinearfluxmethods);
  import_array();
}
