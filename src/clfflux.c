/*
 * XXX
 */

#define PY_ARRAY_UNIQUE_SYMBOL PYBLAW_CLFFLUX_ARRAY_API

#include <stdio.h>
#include <stdlib.h>

#include <Python.h>
#include <numpy/ndarrayobject.h>

/*********************************************************************/

int N, p;
double alpha, *fl, *fr, *dx;

/*********************************************************************/

PyObject *
init_lf_flux(PyObject *self, PyObject *args)
{
  PyObject *dx_py, *fl_py, *fr_py;

  /*
   * parse options
   */
  if (! PyArg_ParseTuple(args, "dOOO", &alpha, &dx_py, &fl_py, &fr_py))
    return NULL;

  if ((PyArray_FLAGS(dx_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "dx is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(dx_py);

  if ((PyArray_FLAGS(fl_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "fl is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(fl_py);

  if ((PyArray_FLAGS(fr_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "fr is not contiguous and/or aligned");
    return NULL;
  }
  Py_INCREF(fr_py);

  /*
   * set globals
   */

  dx = (double *) PyArray_DATA(dx_py);

  fl = (double *) PyArray_DATA(fl_py);
  fr = (double *) PyArray_DATA(fr_py);

  /*
   * done
   */
  Py_INCREF(Py_None);
  return Py_None;
}

/************************************************************************/

void
nflux_lf(double *qm, double *qp, double *fm, double *fp, double *f)
{
  int j;

  for (j=0; j<p; j++)
    f[j] = 0.5 * (fm[j] + fp[j] - alpha * (qp[j] - qm[j]) );
}

PyObject *
lf_flux(PyObject *self, PyObject *args)
{
  long int i;
  PyObject *qm_py, *qp_py, *fm_py, *fp_py, *f_py;
  double *qm, *qp, *fm, *fp, *f;

  int j;

  /*
   * parse options
   */

  if (! PyArg_ParseTuple(args, "OOOOO", &qm_py, &qp_py, &fm_py, &fp_py, &f_py))
    return NULL;

  if ((PyArray_FLAGS(qm_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "qm is not contiguous and/or aligned");
    return NULL;
  }
  qm = (double *) PyArray_DATA(qm_py);

  if ((PyArray_FLAGS(qp_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "qp is not contiguous and/or aligned");
    return NULL;
  }
  qp = (double *) PyArray_DATA(qp_py);

  if ((PyArray_FLAGS(fm_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "fm is not contiguous and/or aligned");
    return NULL;
  }
  fm = (double *) PyArray_DATA(fm_py);

  if ((PyArray_FLAGS(fp_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "fp is not contiguous and/or aligned");
    return NULL;
  }
  fp = (double *) PyArray_DATA(fp_py);

  if ((PyArray_FLAGS(f_py) & NPY_IN_ARRAY) != NPY_IN_ARRAY) {
    PyErr_SetString(PyExc_TypeError, "f is not contiguous and/or aligned");
    return NULL;
  }
  f = (double *) PyArray_DATA(f_py);

  /*
   * compute net flux
   */

  N = PyArray_DIM(f_py, 0);
  p = PyArray_DIM(f_py, 1);

  nflux_lf(qm, qp, fm, fp, fr);

  for (i=1; i<N; i++) {
    for (j=0; j<p; j++)
      fl[j] = fr[j];

    nflux_lf(qm + (i+1)*p, qp + (i+1)*p,
             fm + (i+1)*p, fp + (i+1)*p,
             fr);

    for (j=0; j<p; j++)
      f[i*p+j] = - ( fr[j] - fl[j] ) / dx[i];
  }

  /*
   * done
   */
  Py_INCREF(Py_None);
  return Py_None;
}

/*********************************************************************/

static PyMethodDef clffluxmethods[] = {
    {"init_lf_flux", init_lf_flux, METH_VARARGS, "XXX"},
    {"lf_flux", lf_flux, METH_VARARGS, "XXX"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initclfflux(void)
{
  (void) Py_InitModule("clfflux", clffluxmethods);
  import_array();
}
