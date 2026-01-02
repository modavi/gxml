/*
 * _vec3.c - High-performance 3D vector math C extension for GXML
 * 
 * This module provides:
 *   - Vec3 type: A 3D vector with operator support
 *   - transform_point: Transform a point by a 4x4 matrix
 *   - intersect_line_plane: Line-plane intersection
 *   - distance, length, normalize, dot, cross: Vector operations
 *
 * Build with: python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>

/* ============================================================================
 * Vec3 Type Definition
 * ============================================================================ */

typedef struct {
    PyObject_HEAD
    double x;
    double y;
    double z;
} Vec3Object;

static PyTypeObject Vec3Type;  /* Forward declaration */

/* Helper to create a new Vec3 from components */
static Vec3Object *Vec3_create(double x, double y, double z) {
    Vec3Object *self = (Vec3Object *)Vec3Type.tp_alloc(&Vec3Type, 0);
    if (self != NULL) {
        self->x = x;
        self->y = y;
        self->z = z;
    }
    return self;
}

/* Helper to extract xyz from any sequence-like object */
static int Vec3_extract(PyObject *obj, double *x, double *y, double *z) {
    if (Py_TYPE(obj) == &Vec3Type) {
        Vec3Object *v = (Vec3Object *)obj;
        *x = v->x;
        *y = v->y;
        *z = v->z;
        return 1;
    }
    
    /* Try sequence protocol */
    if (PySequence_Check(obj) && PySequence_Size(obj) >= 3) {
        PyObject *item;
        
        item = PySequence_GetItem(obj, 0);
        if (!item) return 0;
        *x = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) return 0;
        
        item = PySequence_GetItem(obj, 1);
        if (!item) return 0;
        *y = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) return 0;
        
        item = PySequence_GetItem(obj, 2);
        if (!item) return 0;
        *z = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) return 0;
        
        return 1;
    }
    
    PyErr_SetString(PyExc_TypeError, "Expected Vec3 or sequence of 3 numbers");
    return 0;
}

/* Vec3.__new__ */
static PyObject *Vec3_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Vec3Object *self = (Vec3Object *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->x = 0.0;
        self->y = 0.0;
        self->z = 0.0;
    }
    return (PyObject *)self;
}

/* Vec3.__init__ */
static int Vec3_init(Vec3Object *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"x", "y", "z", NULL};
    PyObject *x_obj = NULL;
    double x = 0.0, y = 0.0, z = 0.0;
    
    /* Try parsing as (x, y, z) first */
    if (PyArg_ParseTupleAndKeywords(args, kwds, "|ddd", kwlist, &x, &y, &z)) {
        self->x = x;
        self->y = y;
        self->z = z;
        return 0;
    }
    
    /* Clear error and try parsing as single sequence argument */
    PyErr_Clear();
    if (PyArg_ParseTuple(args, "O", &x_obj)) {
        if (Vec3_extract(x_obj, &x, &y, &z)) {
            self->x = x;
            self->y = y;
            self->z = z;
            return 0;
        }
    }
    
    return -1;
}

/* Vec3.__repr__ */
static PyObject *Vec3_repr(Vec3Object *self) {
    char buf[128];
    snprintf(buf, sizeof(buf), "Vec3(%g, %g, %g)", self->x, self->y, self->z);
    return PyUnicode_FromString(buf);
}

/* Vec3.__getitem__ */
static PyObject *Vec3_getitem(Vec3Object *self, Py_ssize_t i) {
    switch (i) {
        case 0: return PyFloat_FromDouble(self->x);
        case 1: return PyFloat_FromDouble(self->y);
        case 2: return PyFloat_FromDouble(self->z);
        default:
            PyErr_SetString(PyExc_IndexError, "Vec3 index out of range");
            return NULL;
    }
}

/* Vec3 sequence length */
static Py_ssize_t Vec3_length(Vec3Object *self) {
    return 3;
}

/* Sequence methods */
static PySequenceMethods Vec3_as_sequence = {
    .sq_length = (lenfunc)Vec3_length,
    .sq_item = (ssizeargfunc)Vec3_getitem,
};

/* Vec3.__add__ */
static PyObject *Vec3_add(PyObject *a, PyObject *b) {
    double ax, ay, az, bx, by, bz;
    
    if (!Vec3_extract(a, &ax, &ay, &az)) return NULL;
    if (!Vec3_extract(b, &bx, &by, &bz)) return NULL;
    
    return (PyObject *)Vec3_create(ax + bx, ay + by, az + bz);
}

/* Vec3.__sub__ */
static PyObject *Vec3_sub(PyObject *a, PyObject *b) {
    double ax, ay, az, bx, by, bz;
    
    if (!Vec3_extract(a, &ax, &ay, &az)) return NULL;
    if (!Vec3_extract(b, &bx, &by, &bz)) return NULL;
    
    return (PyObject *)Vec3_create(ax - bx, ay - by, az - bz);
}

/* Vec3.__mul__ (scalar) */
static PyObject *Vec3_mul(PyObject *a, PyObject *b) {
    double vx, vy, vz, s;
    
    /* Try Vec3 * scalar */
    if (Py_TYPE(a) == &Vec3Type && PyNumber_Check(b)) {
        Vec3Object *v = (Vec3Object *)a;
        s = PyFloat_AsDouble(b);
        if (PyErr_Occurred()) return NULL;
        return (PyObject *)Vec3_create(v->x * s, v->y * s, v->z * s);
    }
    
    /* Try scalar * Vec3 */
    if (PyNumber_Check(a) && Py_TYPE(b) == &Vec3Type) {
        Vec3Object *v = (Vec3Object *)b;
        s = PyFloat_AsDouble(a);
        if (PyErr_Occurred()) return NULL;
        return (PyObject *)Vec3_create(v->x * s, v->y * s, v->z * s);
    }
    
    Py_RETURN_NOTIMPLEMENTED;
}

/* Vec3.__truediv__ */
static PyObject *Vec3_truediv(PyObject *a, PyObject *b) {
    if (Py_TYPE(a) == &Vec3Type && PyNumber_Check(b)) {
        Vec3Object *v = (Vec3Object *)a;
        double s = PyFloat_AsDouble(b);
        if (PyErr_Occurred()) return NULL;
        if (s == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
            return NULL;
        }
        double inv_s = 1.0 / s;
        return (PyObject *)Vec3_create(v->x * inv_s, v->y * inv_s, v->z * inv_s);
    }
    Py_RETURN_NOTIMPLEMENTED;
}

/* Vec3.__neg__ */
static PyObject *Vec3_neg(Vec3Object *self) {
    return (PyObject *)Vec3_create(-self->x, -self->y, -self->z);
}

/* Number methods */
static PyNumberMethods Vec3_as_number = {
    .nb_add = Vec3_add,
    .nb_subtract = Vec3_sub,
    .nb_multiply = Vec3_mul,
    .nb_negative = (unaryfunc)Vec3_neg,
    .nb_true_divide = Vec3_truediv,
};

/* Vec3.dot(other) */
static PyObject *Vec3_dot(Vec3Object *self, PyObject *args) {
    PyObject *other;
    double ox, oy, oz;
    
    if (!PyArg_ParseTuple(args, "O", &other)) return NULL;
    if (!Vec3_extract(other, &ox, &oy, &oz)) return NULL;
    
    return PyFloat_FromDouble(self->x * ox + self->y * oy + self->z * oz);
}

/* Vec3.cross(other) */
static PyObject *Vec3_cross(Vec3Object *self, PyObject *args) {
    PyObject *other;
    double ox, oy, oz;
    
    if (!PyArg_ParseTuple(args, "O", &other)) return NULL;
    if (!Vec3_extract(other, &ox, &oy, &oz)) return NULL;
    
    return (PyObject *)Vec3_create(
        self->y * oz - self->z * oy,
        self->z * ox - self->x * oz,
        self->x * oy - self->y * ox
    );
}

/* Vec3.length() */
static PyObject *Vec3_length_method(Vec3Object *self, PyObject *Py_UNUSED(ignored)) {
    return PyFloat_FromDouble(sqrt(self->x * self->x + self->y * self->y + self->z * self->z));
}

/* Vec3.length_sq() */
static PyObject *Vec3_length_sq(Vec3Object *self, PyObject *Py_UNUSED(ignored)) {
    return PyFloat_FromDouble(self->x * self->x + self->y * self->y + self->z * self->z);
}

/* Vec3.normalized() */
static PyObject *Vec3_normalized(Vec3Object *self, PyObject *Py_UNUSED(ignored)) {
    double mag = sqrt(self->x * self->x + self->y * self->y + self->z * self->z);
    if (mag < 1e-10) {
        return (PyObject *)Vec3_create(0.0, 0.0, 0.0);
    }
    double inv_mag = 1.0 / mag;
    return (PyObject *)Vec3_create(self->x * inv_mag, self->y * inv_mag, self->z * inv_mag);
}

/* Vec3.to_tuple() */
static PyObject *Vec3_to_tuple(Vec3Object *self, PyObject *Py_UNUSED(ignored)) {
    return Py_BuildValue("(ddd)", self->x, self->y, self->z);
}

/* Vec3 methods */
static PyMethodDef Vec3_methods[] = {
    {"dot", (PyCFunction)Vec3_dot, METH_VARARGS, "Dot product with another vector"},
    {"cross", (PyCFunction)Vec3_cross, METH_VARARGS, "Cross product with another vector"},
    {"length", (PyCFunction)Vec3_length_method, METH_NOARGS, "Vector length/magnitude"},
    {"length_sq", (PyCFunction)Vec3_length_sq, METH_NOARGS, "Squared length"},
    {"normalized", (PyCFunction)Vec3_normalized, METH_NOARGS, "Return normalized copy"},
    {"to_tuple", (PyCFunction)Vec3_to_tuple, METH_NOARGS, "Convert to tuple"},
    {NULL}
};

/* Vec3 member definitions */
static PyMemberDef Vec3_members[] = {
    {"x", T_DOUBLE, offsetof(Vec3Object, x), 0, "X component"},
    {"y", T_DOUBLE, offsetof(Vec3Object, y), 0, "Y component"},
    {"z", T_DOUBLE, offsetof(Vec3Object, z), 0, "Z component"},
    {NULL}
};

/* Vec3 type definition */
static PyTypeObject Vec3Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_vec3.Vec3",
    .tp_doc = "3D vector with fast math operations",
    .tp_basicsize = sizeof(Vec3Object),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Vec3_new,
    .tp_init = (initproc)Vec3_init,
    .tp_repr = (reprfunc)Vec3_repr,
    .tp_as_number = &Vec3_as_number,
    .tp_as_sequence = &Vec3_as_sequence,
    .tp_methods = Vec3_methods,
    .tp_members = Vec3_members,
};


/* ============================================================================
 * Module-Level Functions
 * ============================================================================ */

/* transform_point(point, matrix) - Transform a 3D point by a 4x4 matrix */
static PyObject *vec3_transform_point(PyObject *self, PyObject *args) {
    PyObject *point_obj, *matrix_obj;
    double x, y, z;
    double m[4][4];
    
    if (!PyArg_ParseTuple(args, "OO", &point_obj, &matrix_obj)) return NULL;
    
    /* Extract point */
    if (!Vec3_extract(point_obj, &x, &y, &z)) return NULL;
    
    /* Extract 4x4 matrix - support both numpy arrays and nested lists */
    for (int i = 0; i < 4; i++) {
        PyObject *row = PySequence_GetItem(matrix_obj, i);
        if (!row) return NULL;
        
        for (int j = 0; j < 4; j++) {
            PyObject *item = PySequence_GetItem(row, j);
            if (!item) {
                Py_DECREF(row);
                return NULL;
            }
            m[i][j] = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) {
                Py_DECREF(row);
                return NULL;
            }
        }
        Py_DECREF(row);
    }
    
    /* Row-vector * matrix multiplication */
    double w = x * m[0][3] + y * m[1][3] + z * m[2][3] + m[3][3];
    if (fabs(w) < 1e-10) {
        return (PyObject *)Vec3_create(0.0, 0.0, 0.0);
    }
    
    double inv_w = 1.0 / w;
    return (PyObject *)Vec3_create(
        (x * m[0][0] + y * m[1][0] + z * m[2][0] + m[3][0]) * inv_w,
        (x * m[0][1] + y * m[1][1] + z * m[2][1] + m[3][1]) * inv_w,
        (x * m[0][2] + y * m[1][2] + z * m[2][2] + m[3][2]) * inv_w
    );
}

/* intersect_line_plane(line_point, line_dir, plane_point, plane_normal) */
static PyObject *vec3_intersect_line_plane(PyObject *self, PyObject *args) {
    PyObject *lp_obj, *ld_obj, *pp_obj, *pn_obj;
    double lpx, lpy, lpz;
    double ldx, ldy, ldz;
    double ppx, ppy, ppz;
    double pnx, pny, pnz;
    
    if (!PyArg_ParseTuple(args, "OOOO", &lp_obj, &ld_obj, &pp_obj, &pn_obj)) return NULL;
    
    if (!Vec3_extract(lp_obj, &lpx, &lpy, &lpz)) return NULL;
    if (!Vec3_extract(ld_obj, &ldx, &ldy, &ldz)) return NULL;
    if (!Vec3_extract(pp_obj, &ppx, &ppy, &ppz)) return NULL;
    if (!Vec3_extract(pn_obj, &pnx, &pny, &pnz)) return NULL;
    
    double denom = ldx * pnx + ldy * pny + ldz * pnz;
    
    if (fabs(denom) < 1e-10) {
        Py_RETURN_NONE;
    }
    
    double t = ((ppx - lpx) * pnx + (ppy - lpy) * pny + (ppz - lpz) * pnz) / denom;
    
    return (PyObject *)Vec3_create(
        lpx + t * ldx,
        lpy + t * ldy,
        lpz + t * ldz
    );
}

/* distance(p1, p2) */
static PyObject *vec3_distance(PyObject *self, PyObject *args) {
    PyObject *p1_obj, *p2_obj;
    double x1, y1, z1, x2, y2, z2;
    
    if (!PyArg_ParseTuple(args, "OO", &p1_obj, &p2_obj)) return NULL;
    if (!Vec3_extract(p1_obj, &x1, &y1, &z1)) return NULL;
    if (!Vec3_extract(p2_obj, &x2, &y2, &z2)) return NULL;
    
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;
    
    return PyFloat_FromDouble(sqrt(dx*dx + dy*dy + dz*dz));
}

/* length(v) */
static PyObject *vec3_length(PyObject *self, PyObject *args) {
    PyObject *v_obj;
    double x, y, z;
    
    if (!PyArg_ParseTuple(args, "O", &v_obj)) return NULL;
    if (!Vec3_extract(v_obj, &x, &y, &z)) return NULL;
    
    return PyFloat_FromDouble(sqrt(x*x + y*y + z*z));
}

/* normalize(v) */
static PyObject *vec3_normalize(PyObject *self, PyObject *args) {
    PyObject *v_obj;
    double x, y, z;
    
    if (!PyArg_ParseTuple(args, "O", &v_obj)) return NULL;
    if (!Vec3_extract(v_obj, &x, &y, &z)) return NULL;
    
    double mag = sqrt(x*x + y*y + z*z);
    if (mag < 1e-10) {
        PyErr_SetString(PyExc_ValueError, "Cannot normalize a zero vector");
        return NULL;
    }
    
    double inv_mag = 1.0 / mag;
    return Py_BuildValue("(ddd)", x * inv_mag, y * inv_mag, z * inv_mag);
}

/* dot(a, b) */
static PyObject *vec3_dot(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj;
    double ax, ay, az, bx, by, bz;
    
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    if (!Vec3_extract(a_obj, &ax, &ay, &az)) return NULL;
    if (!Vec3_extract(b_obj, &bx, &by, &bz)) return NULL;
    
    return PyFloat_FromDouble(ax*bx + ay*by + az*bz);
}

/* cross(a, b) */
static PyObject *vec3_cross(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj;
    double ax, ay, az, bx, by, bz;
    
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    if (!Vec3_extract(a_obj, &ax, &ay, &az)) return NULL;
    if (!Vec3_extract(b_obj, &bx, &by, &bz)) return NULL;
    
    return Py_BuildValue("(ddd)",
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx
    );
}

/* lerp(t, a, b) - Linear interpolation between two Vec3s */
static PyObject *vec3_lerp(PyObject *self, PyObject *args) {
    double t;
    PyObject *a_obj, *b_obj;
    double ax, ay, az, bx, by, bz;
    
    if (!PyArg_ParseTuple(args, "dOO", &t, &a_obj, &b_obj)) return NULL;
    if (!Vec3_extract(a_obj, &ax, &ay, &az)) return NULL;
    if (!Vec3_extract(b_obj, &bx, &by, &bz)) return NULL;
    
    double one_minus_t = 1.0 - t;
    return (PyObject *)Vec3_create(
        one_minus_t * ax + t * bx,
        one_minus_t * ay + t * by,
        one_minus_t * az + t * bz
    );
}

/* mat4_invert(matrix) - Invert a 4x4 matrix, returns flat tuple of 16 values (row-major) */
static PyObject *vec3_mat4_invert(PyObject *self, PyObject *args) {
    PyObject *mat_obj;
    double m[16];  /* Input matrix */
    double inv[16]; /* Inverted matrix */
    
    if (!PyArg_ParseTuple(args, "O", &mat_obj)) return NULL;
    
    /* Extract 4x4 matrix from nested sequence */
    if (!PySequence_Check(mat_obj) || PySequence_Size(mat_obj) != 4) {
        PyErr_SetString(PyExc_TypeError, "Expected 4x4 matrix (sequence of 4 sequences)");
        return NULL;
    }
    
    for (int row = 0; row < 4; row++) {
        PyObject *row_obj = PySequence_GetItem(mat_obj, row);
        if (!row_obj || !PySequence_Check(row_obj) || PySequence_Size(row_obj) != 4) {
            Py_XDECREF(row_obj);
            PyErr_SetString(PyExc_TypeError, "Expected 4x4 matrix (sequence of 4 sequences)");
            return NULL;
        }
        for (int col = 0; col < 4; col++) {
            PyObject *val = PySequence_GetItem(row_obj, col);
            if (!val) {
                Py_DECREF(row_obj);
                return NULL;
            }
            m[row * 4 + col] = PyFloat_AsDouble(val);
            Py_DECREF(val);
            if (PyErr_Occurred()) {
                Py_DECREF(row_obj);
                return NULL;
            }
        }
        Py_DECREF(row_obj);
    }
    
    /* Compute 4x4 matrix inverse using cofactor expansion */
    /* This is the standard formula - compute cofactors and divide by determinant */
    
    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + 
             m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - 
             m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + 
             m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - 
              m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    
    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - 
             m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + 
             m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - 
             m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + 
              m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    
    inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + 
             m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - 
             m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + 
              m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - 
              m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    
    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - 
             m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + 
             m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - 
              m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + 
              m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
    
    double det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    
    if (fabs(det) < 1e-15) {
        PyErr_SetString(PyExc_ValueError, "Matrix is singular and cannot be inverted");
        return NULL;
    }
    
    double inv_det = 1.0 / det;
    
    /* Build nested tuple result (4x4 matrix as tuple of tuples) */
    return Py_BuildValue("((dddd)(dddd)(dddd)(dddd))",
        inv[0]*inv_det, inv[1]*inv_det, inv[2]*inv_det, inv[3]*inv_det,
        inv[4]*inv_det, inv[5]*inv_det, inv[6]*inv_det, inv[7]*inv_det,
        inv[8]*inv_det, inv[9]*inv_det, inv[10]*inv_det, inv[11]*inv_det,
        inv[12]*inv_det, inv[13]*inv_det, inv[14]*inv_det, inv[15]*inv_det
    );
}

/* find_interpolated_point(point, p1, p2) - Find interpolation value of point on line segment */
static PyObject *vec3_find_interpolated_point(PyObject *self, PyObject *args) {
    PyObject *point_obj, *p1_obj, *p2_obj;
    double px, py, pz, p1x, p1y, p1z, p2x, p2y, p2z;
    
    if (!PyArg_ParseTuple(args, "OOO", &point_obj, &p1_obj, &p2_obj)) return NULL;
    
    if (!Vec3_extract(point_obj, &px, &py, &pz)) return NULL;
    if (!Vec3_extract(p1_obj, &p1x, &p1y, &p1z)) return NULL;
    if (!Vec3_extract(p2_obj, &p2x, &p2y, &p2z)) return NULL;
    
    /* segment_vector = p2 - p1 */
    double sx = p2x - p1x;
    double sy = p2y - p1y;
    double sz = p2z - p1z;
    
    /* point_vector = point - p1 */
    double vx = px - p1x;
    double vy = py - p1y;
    double vz = pz - p1z;
    
    /* dot(point_vector, segment_vector) / dot(segment_vector, segment_vector) */
    double denom = sx*sx + sy*sy + sz*sz;
    if (fabs(denom) < 1e-15) {
        return PyFloat_FromDouble(0.0);
    }
    
    double numer = vx*sx + vy*sy + vz*sz;
    return PyFloat_FromDouble(numer / denom);
}

/* Module method definitions */
static PyMethodDef vec3_methods[] = {
    {"transform_point", vec3_transform_point, METH_VARARGS, 
     "Transform a 3D point by a 4x4 matrix"},
    {"intersect_line_plane", vec3_intersect_line_plane, METH_VARARGS,
     "Intersect a line with a plane, returns Vec3 or None"},
    {"distance", vec3_distance, METH_VARARGS,
     "Distance between two points"},
    {"length", vec3_length, METH_VARARGS,
     "Length of a vector"},
    {"normalize", vec3_normalize, METH_VARARGS,
     "Normalize a vector, returns tuple"},
    {"dot", vec3_dot, METH_VARARGS,
     "Dot product of two vectors"},
    {"cross", vec3_cross, METH_VARARGS,
     "Cross product of two vectors, returns tuple"},
    {"lerp", vec3_lerp, METH_VARARGS,
     "Linear interpolation: lerp(t, a, b) returns Vec3"},
    {"mat4_invert", vec3_mat4_invert, METH_VARARGS,
     "Invert a 4x4 matrix, returns tuple of tuples"},
    {"find_interpolated_point", vec3_find_interpolated_point, METH_VARARGS,
     "Find interpolation value of point on line segment"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef vec3module = {
    PyModuleDef_HEAD_INIT,
    "_vec3",
    "High-performance 3D vector math for GXML",
    -1,
    vec3_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit__vec3(void) {
    PyObject *m;
    
    if (PyType_Ready(&Vec3Type) < 0)
        return NULL;
    
    m = PyModule_Create(&vec3module);
    if (m == NULL)
        return NULL;
    
    Py_INCREF(&Vec3Type);
    if (PyModule_AddObject(m, "Vec3", (PyObject *)&Vec3Type) < 0) {
        Py_DECREF(&Vec3Type);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}
