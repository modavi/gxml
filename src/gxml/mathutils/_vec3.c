/*
 * _vec3.c - High-performance 3D vector math C extension for GXML
 * 
 * This module provides:
 *   - Vec3 type: A 3D vector with operator support
 *   - Mat4 type: A 4x4 transformation matrix with SIMD-optimized operations
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

/* SIMD Support Detection and Includes */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define USE_SSE 1
    #if defined(__AVX__)
        #define USE_AVX 1
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define USE_NEON 1
#endif

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
 * Mat4 Type Definition - 4x4 Matrix stored as contiguous C doubles
 * For optimal SIMD performance, ensure 32-byte alignment for AVX operations
 * ============================================================================ */

typedef struct {
    PyObject_HEAD
#if defined(USE_AVX)
    __attribute__((aligned(32))) double m[16];  /* 32-byte aligned for AVX */
#elif defined(USE_SSE) || defined(USE_NEON)
    __attribute__((aligned(16))) double m[16];  /* 16-byte aligned for SSE/NEON */
#else
    double m[16];  /* Row-major: m[row*4 + col] */
#endif
} Mat4Object;

static PyTypeObject Mat4Type;  /* Forward declaration */

/* Helper to create identity matrix */
static Mat4Object *Mat4_create_identity(void) {
    Mat4Object *self = (Mat4Object *)Mat4Type.tp_alloc(&Mat4Type, 0);
    if (self != NULL) {
        for (int i = 0; i < 16; i++) self->m[i] = 0.0;
        self->m[0] = self->m[5] = self->m[10] = self->m[15] = 1.0;
    }
    return self;
}

/* Helper to create Mat4 from 16 doubles */
static Mat4Object *Mat4_create_from_array(const double *arr) {
    Mat4Object *self = (Mat4Object *)Mat4Type.tp_alloc(&Mat4Type, 0);
    if (self != NULL) {
        for (int i = 0; i < 16; i++) self->m[i] = arr[i];
    }
    return self;
}

/* Helper to extract matrix data from Mat4 or nested sequence */
static int Mat4_extract(PyObject *obj, double *out) {
    if (Py_TYPE(obj) == &Mat4Type) {
        Mat4Object *mat = (Mat4Object *)obj;
        for (int i = 0; i < 16; i++) out[i] = mat->m[i];
        return 1;
    }
    
    /* Try nested sequence (tuple of tuples or list of lists) */
    if (PySequence_Check(obj) && PySequence_Size(obj) == 4) {
        for (int row = 0; row < 4; row++) {
            PyObject *row_obj = PySequence_GetItem(obj, row);
            if (!row_obj || !PySequence_Check(row_obj) || PySequence_Size(row_obj) != 4) {
                Py_XDECREF(row_obj);
                PyErr_SetString(PyExc_TypeError, "Expected Mat4 or 4x4 nested sequence");
                return 0;
            }
            for (int col = 0; col < 4; col++) {
                PyObject *val = PySequence_GetItem(row_obj, col);
                if (!val) {
                    Py_DECREF(row_obj);
                    return 0;
                }
                out[row * 4 + col] = PyFloat_AsDouble(val);
                Py_DECREF(val);
                if (PyErr_Occurred()) {
                    Py_DECREF(row_obj);
                    return 0;
                }
            }
            Py_DECREF(row_obj);
        }
        return 1;
    }
    
    PyErr_SetString(PyExc_TypeError, "Expected Mat4 or 4x4 nested sequence");
    return 0;
}

/* Mat4.__new__ */
static PyObject *Mat4_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Mat4Object *self = (Mat4Object *)type->tp_alloc(type, 0);
    if (self != NULL) {
        /* Initialize to identity */
        for (int i = 0; i < 16; i++) self->m[i] = 0.0;
        self->m[0] = self->m[5] = self->m[10] = self->m[15] = 1.0;
    }
    return (PyObject *)self;
}

/* Mat4.__init__ - Accept nested sequence or nothing (identity) */
static int Mat4_init(Mat4Object *self, PyObject *args, PyObject *kwds) {
    PyObject *data = NULL;
    
    if (!PyArg_ParseTuple(args, "|O", &data)) return -1;
    
    if (data == NULL) {
        /* No args = identity matrix */
        for (int i = 0; i < 16; i++) self->m[i] = 0.0;
        self->m[0] = self->m[5] = self->m[10] = self->m[15] = 1.0;
        return 0;
    }
    
    if (!Mat4_extract(data, self->m)) return -1;
    return 0;
}

/* Mat4.__repr__ */
static PyObject *Mat4_repr(Mat4Object *self) {
    char buf[512];
    snprintf(buf, sizeof(buf),
        "Mat4([\n"
        "  [%.4g, %.4g, %.4g, %.4g],\n"
        "  [%.4g, %.4g, %.4g, %.4g],\n"
        "  [%.4g, %.4g, %.4g, %.4g],\n"
        "  [%.4g, %.4g, %.4g, %.4g]])",
        self->m[0], self->m[1], self->m[2], self->m[3],
        self->m[4], self->m[5], self->m[6], self->m[7],
        self->m[8], self->m[9], self->m[10], self->m[11],
        self->m[12], self->m[13], self->m[14], self->m[15]);
    return PyUnicode_FromString(buf);
}

/* Mat4.__getitem__ - returns a row as tuple */
static PyObject *Mat4_getitem(Mat4Object *self, Py_ssize_t i) {
    if (i < 0 || i > 3) {
        PyErr_SetString(PyExc_IndexError, "Mat4 row index out of range (0-3)");
        return NULL;
    }
    int base = i * 4;
    return Py_BuildValue("(dddd)", self->m[base], self->m[base+1], 
                         self->m[base+2], self->m[base+3]);
}

/* Mat4 sequence length */
static Py_ssize_t Mat4_length(Mat4Object *self) {
    return 4;
}

/* Sequence methods for Mat4 */
static PySequenceMethods Mat4_as_sequence = {
    .sq_length = (lenfunc)Mat4_length,
    .sq_item = (ssizeargfunc)Mat4_getitem,
};

/* Mat4.__matmul__ - Matrix multiplication */
static PyObject *Mat4_matmul(PyObject *a_obj, PyObject *b_obj) {
    double a[16], b[16], r[16];
    
    if (!Mat4_extract(a_obj, a)) return NULL;
    if (!Mat4_extract(b_obj, b)) return NULL;
    
    /* r = a @ b */
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            r[row*4 + col] = 
                a[row*4 + 0] * b[0*4 + col] +
                a[row*4 + 1] * b[1*4 + col] +
                a[row*4 + 2] * b[2*4 + col] +
                a[row*4 + 3] * b[3*4 + col];
        }
    }
    
    return (PyObject *)Mat4_create_from_array(r);
}

/* Number methods for @ operator */
static PyNumberMethods Mat4_as_number = {
    .nb_matrix_multiply = Mat4_matmul,
};

/* ============================================================================
 * SIMD-optimized matrix-vector multiplication helpers
 * ============================================================================ */

#ifdef USE_SSE
/* SSE2 optimized 4x4 matrix * vec4 multiplication
 * Matrix is row-major, computes result = M * [x, y, z, 1]
 * Returns result in result[0]=x, result[1]=y, result[2]=z, result[3]=w
 */
static inline void mat4_transform_point_sse(const double *m, double x, double y, double z,
                                            double *result) {
    /* Load point as [x, y, z, 1] and broadcast each component */
    __m128d vx = _mm_set1_pd(x);
    __m128d vy = _mm_set1_pd(y);
    __m128d vz = _mm_set1_pd(z);
    __m128d vw = _mm_set1_pd(1.0);
    
    /* Process two output components at a time (SSE2 handles 2 doubles) */
    /* Components 0-1 (result_x, result_y) */
    __m128d col0_01 = _mm_loadu_pd(&m[0]);   /* m[0], m[1] */
    __m128d col1_01 = _mm_loadu_pd(&m[4]);   /* m[4], m[5] */
    __m128d col2_01 = _mm_loadu_pd(&m[8]);   /* m[8], m[9] */
    __m128d col3_01 = _mm_loadu_pd(&m[12]);  /* m[12], m[13] */
    
    __m128d res01 = _mm_add_pd(
        _mm_add_pd(_mm_mul_pd(vx, col0_01), _mm_mul_pd(vy, col1_01)),
        _mm_add_pd(_mm_mul_pd(vz, col2_01), _mm_mul_pd(vw, col3_01))
    );
    
    /* Components 2-3 (result_z, result_w) */
    __m128d col0_23 = _mm_loadu_pd(&m[2]);   /* m[2], m[3] */
    __m128d col1_23 = _mm_loadu_pd(&m[6]);   /* m[6], m[7] */
    __m128d col2_23 = _mm_loadu_pd(&m[10]);  /* m[10], m[11] */
    __m128d col3_23 = _mm_loadu_pd(&m[14]);  /* m[14], m[15] */
    
    __m128d res23 = _mm_add_pd(
        _mm_add_pd(_mm_mul_pd(vx, col0_23), _mm_mul_pd(vy, col1_23)),
        _mm_add_pd(_mm_mul_pd(vz, col2_23), _mm_mul_pd(vw, col3_23))
    );
    
    /* Store results to contiguous array */
    _mm_storeu_pd(&result[0], res01);  /* stores result[0] and result[1] */
    _mm_storeu_pd(&result[2], res23);  /* stores result[2] and result[3] */
}
#endif

/* Scalar fallback for matrix-vector multiply */
static inline void mat4_transform_point_scalar(const double *m, double x, double y, double z,
                                               double *result) {
    result[0] = x * m[0] + y * m[4] + z * m[8] + m[12];
    result[1] = x * m[1] + y * m[5] + z * m[9] + m[13];
    result[2] = x * m[2] + y * m[6] + z * m[10] + m[14];
    result[3] = x * m[3] + y * m[7] + z * m[11] + m[15];
}

/* Mat4.transform_point(point) - Transform a point, returns Vec3 */
static PyObject *Mat4_transform_point(Mat4Object *self, PyObject *args) {
    PyObject *point_obj;
    double x, y, z;
    
    if (!PyArg_ParseTuple(args, "O", &point_obj)) return NULL;
    if (!Vec3_extract(point_obj, &x, &y, &z)) return NULL;
    
    double result[4];
    
#ifdef USE_SSE
    mat4_transform_point_sse(self->m, x, y, z, result);
#else
    mat4_transform_point_scalar(self->m, x, y, z, result);
#endif
    
    double rw = result[3];
    if (fabs(rw) < 1e-10) {
        return (PyObject *)Vec3_create(0.0, 0.0, 0.0);
    }
    
    double inv_w = 1.0 / rw;
    return (PyObject *)Vec3_create(result[0] * inv_w, result[1] * inv_w, result[2] * inv_w);
}

/* Mat4.bilinear_transform(t, s, z_offset, p0, p1, p2, p3) 
 * Combined bilinear interpolation + matrix transform in one call.
 * Uses SIMD for both bilinear and matrix multiply when available.
 * Returns Vec3.
 */
static PyObject *Mat4_bilinear_transform(Mat4Object *self, PyObject *args) {
    double t, s, z_offset;
    PyObject *p0_obj, *p1_obj, *p2_obj, *p3_obj;
    double q0x, q0y, q0z, q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;
    
    if (!PyArg_ParseTuple(args, "dddOOOO", &t, &s, &z_offset, 
                          &p0_obj, &p1_obj, &p2_obj, &p3_obj)) return NULL;
    
    if (!Vec3_extract(p0_obj, &q0x, &q0y, &q0z)) return NULL;
    if (!Vec3_extract(p1_obj, &q1x, &q1y, &q1z)) return NULL;
    if (!Vec3_extract(p2_obj, &q2x, &q2y, &q2z)) return NULL;
    if (!Vec3_extract(p3_obj, &q3x, &q3y, &q3z)) return NULL;
    
    double ix, iy, iz;
    
#ifdef USE_SSE
    /* SIMD bilinear interpolation - process x,y pairs in parallel */
    __m128d vt = _mm_set1_pd(t);
    __m128d vs = _mm_set1_pd(s);
    
    /* l1 = p0 + t * (p1 - p0) for x,y */
    __m128d p0_xy = _mm_set_pd(q0y, q0x);
    __m128d p1_xy = _mm_set_pd(q1y, q1x);
    __m128d l1_xy = _mm_add_pd(p0_xy, _mm_mul_pd(vt, _mm_sub_pd(p1_xy, p0_xy)));
    
    /* l2 = p3 + t * (p2 - p3) for x,y */
    __m128d p3_xy = _mm_set_pd(q3y, q3x);
    __m128d p2_xy = _mm_set_pd(q2y, q2x);
    __m128d l2_xy = _mm_add_pd(p3_xy, _mm_mul_pd(vt, _mm_sub_pd(p2_xy, p3_xy)));
    
    /* result = l1 + s * (l2 - l1) for x,y */
    __m128d res_xy = _mm_add_pd(l1_xy, _mm_mul_pd(vs, _mm_sub_pd(l2_xy, l1_xy)));
    
    double xy[2];
    _mm_storeu_pd(xy, res_xy);
    ix = xy[0];
    iy = xy[1];
    
    /* z component scalar (only one value, not worth SIMD) */
    double l1z = q0z + t * (q1z - q0z);
    double l2z = q3z + t * (q2z - q3z);
    iz = l1z + s * (l2z - l1z) + z_offset;
#else
    /* Scalar bilinear interpolation */
    double l1x = q0x + t * (q1x - q0x);
    double l1y = q0y + t * (q1y - q0y);
    double l1z = q0z + t * (q1z - q0z);
    
    double l2x = q3x + t * (q2x - q3x);
    double l2y = q3y + t * (q2y - q3y);
    double l2z = q3z + t * (q2z - q3z);
    
    ix = l1x + s * (l2x - l1x);
    iy = l1y + s * (l2y - l1y);
    iz = l1z + s * (l2z - l1z) + z_offset;
#endif
    
    /* Matrix transform with SIMD */
    double result[4];
    
#ifdef USE_SSE
    mat4_transform_point_sse(self->m, ix, iy, iz, result);
#else
    mat4_transform_point_scalar(self->m, ix, iy, iz, result);
#endif
    
    if (fabs(result[3]) < 1e-10) {
        return (PyObject *)Vec3_create(0.0, 0.0, 0.0);
    }
    
    double inv_w = 1.0 / result[3];
    return (PyObject *)Vec3_create(result[0] * inv_w, result[1] * inv_w, result[2] * inv_w);
}

/* Mat4.transform_direction(dir) - Transform a direction (no translation), returns Vec3 */
static PyObject *Mat4_transform_direction(Mat4Object *self, PyObject *args) {
    PyObject *dir_obj;
    double x, y, z;
    
    if (!PyArg_ParseTuple(args, "O", &dir_obj)) return NULL;
    if (!Vec3_extract(dir_obj, &x, &y, &z)) return NULL;
    
    double *m = self->m;
    return (PyObject *)Vec3_create(
        x * m[0] + y * m[4] + z * m[8],
        x * m[1] + y * m[5] + z * m[9],
        x * m[2] + y * m[6] + z * m[10]
    );
}

/* Mat4.inverse() - Return inverted matrix */
static PyObject *Mat4_inverse(Mat4Object *self, PyObject *Py_UNUSED(ignored)) {
    double *m = self->m;
    double inv[16];
    
    /* Compute 4x4 matrix inverse using cofactor expansion */
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
    for (int i = 0; i < 16; i++) inv[i] *= inv_det;
    
    return (PyObject *)Mat4_create_from_array(inv);
}

/* Mat4.to_tuple() - Convert to nested tuple */
static PyObject *Mat4_to_tuple(Mat4Object *self, PyObject *Py_UNUSED(ignored)) {
    double *m = self->m;
    return Py_BuildValue("((dddd)(dddd)(dddd)(dddd))",
        m[0], m[1], m[2], m[3],
        m[4], m[5], m[6], m[7],
        m[8], m[9], m[10], m[11],
        m[12], m[13], m[14], m[15]
    );
}

/* Mat4.batch_transform_points(points) - Transform multiple points at once */
static PyObject *Mat4_batch_transform_points(Mat4Object *self, PyObject *args) {
    PyObject *points_obj;
    
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return NULL;
    
    Py_ssize_t n = PySequence_Size(points_obj);
    if (n < 0) return NULL;
    
    PyObject *result = PyList_New(n);
    if (!result) return NULL;
    
    double *m = self->m;
    
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *point = PySequence_GetItem(points_obj, i);
        if (!point) {
            Py_DECREF(result);
            return NULL;
        }
        
        double x, y, z;
        if (!Vec3_extract(point, &x, &y, &z)) {
            Py_DECREF(point);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(point);
        
        double w = x * m[3] + y * m[7] + z * m[11] + m[15];
        double inv_w = (fabs(w) < 1e-10) ? 0.0 : (1.0 / w);
        
        PyObject *transformed = Py_BuildValue("(ddd)",
            (x * m[0] + y * m[4] + z * m[8] + m[12]) * inv_w,
            (x * m[1] + y * m[5] + z * m[9] + m[13]) * inv_w,
            (x * m[2] + y * m[6] + z * m[10] + m[14]) * inv_w
        );
        
        if (!transformed) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, transformed);
    }
    
    return result;
}

/* Mat4.batch_bilinear_transform(points_with_offsets, quad_points) - Bilinear + transform */
static PyObject *Mat4_batch_bilinear_transform(Mat4Object *self, PyObject *args) {
    PyObject *points_obj, *quad_obj;
    double q0x, q0y, q0z, q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;
    
    if (!PyArg_ParseTuple(args, "OO", &points_obj, &quad_obj)) return NULL;
    
    /* Extract quad points */
    PyObject *p0 = PySequence_GetItem(quad_obj, 0);
    PyObject *p1 = PySequence_GetItem(quad_obj, 1);
    PyObject *p2 = PySequence_GetItem(quad_obj, 2);
    PyObject *p3 = PySequence_GetItem(quad_obj, 3);
    if (!p0 || !p1 || !p2 || !p3) {
        Py_XDECREF(p0); Py_XDECREF(p1); Py_XDECREF(p2); Py_XDECREF(p3);
        return NULL;
    }
    if (!Vec3_extract(p0, &q0x, &q0y, &q0z) ||
        !Vec3_extract(p1, &q1x, &q1y, &q1z) ||
        !Vec3_extract(p2, &q2x, &q2y, &q2z) ||
        !Vec3_extract(p3, &q3x, &q3y, &q3z)) {
        Py_DECREF(p0); Py_DECREF(p1); Py_DECREF(p2); Py_DECREF(p3);
        return NULL;
    }
    Py_DECREF(p0); Py_DECREF(p1); Py_DECREF(p2); Py_DECREF(p3);
    
    Py_ssize_t n = PySequence_Size(points_obj);
    PyObject *result = PyList_New(n);
    if (!result) return NULL;
    
    double *m = self->m;
    
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *point_tuple = PySequence_GetItem(points_obj, i);
        if (!point_tuple) { Py_DECREF(result); return NULL; }
        
        double t, s, z_offset;
        if (!Vec3_extract(point_tuple, &t, &s, &z_offset)) {
            Py_DECREF(point_tuple);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(point_tuple);
        
        /* Bilinear interpolation */
        double l1x = q0x + t * (q1x - q0x);
        double l1y = q0y + t * (q1y - q0y);
        double l1z = q0z + t * (q1z - q0z);
        
        double l2x = q3x + t * (q2x - q3x);
        double l2y = q3y + t * (q2y - q3y);
        double l2z = q3z + t * (q2z - q3z);
        
        double ix = l1x + s * (l2x - l1x);
        double iy = l1y + s * (l2y - l1y);
        double iz = l1z + s * (l2z - l1z) + z_offset;
        
        /* Matrix transform */
        double w = ix * m[3] + iy * m[7] + iz * m[11] + m[15];
        double inv_w = (fabs(w) < 1e-10) ? 0.0 : (1.0 / w);
        
        PyObject *transformed = Py_BuildValue("(ddd)",
            (ix * m[0] + iy * m[4] + iz * m[8] + m[12]) * inv_w,
            (ix * m[1] + iy * m[5] + iz * m[9] + m[13]) * inv_w,
            (ix * m[2] + iy * m[6] + iz * m[10] + m[14]) * inv_w
        );
        
        if (!transformed) { Py_DECREF(result); return NULL; }
        PyList_SET_ITEM(result, i, transformed);
    }
    
    return result;
}

/* ============== In-place Mat4 methods ============== */

/* Mat4.set_identity() - Reset to identity matrix in-place */
static PyObject *Mat4_set_identity(Mat4Object *self, PyObject *Py_UNUSED(ignored)) {
    for (int i = 0; i < 16; i++) self->m[i] = 0.0;
    self->m[0] = self->m[5] = self->m[10] = self->m[15] = 1.0;
    Py_RETURN_NONE;
}

/* Mat4.set_from(other) - Copy values from another matrix in-place */
static PyObject *Mat4_set_from(Mat4Object *self, PyObject *args) {
    PyObject *other;
    if (!PyArg_ParseTuple(args, "O", &other)) return NULL;
    
    if (!Mat4_extract(other, self->m)) return NULL;
    Py_RETURN_NONE;
}

/* Mat4.multiply_into(a, b) - Compute a @ b and store result in self */
static PyObject *Mat4_multiply_into(Mat4Object *self, PyObject *args) {
    PyObject *a_obj, *b_obj;
    double a[16], b[16];
    
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    if (!Mat4_extract(a_obj, a)) return NULL;
    if (!Mat4_extract(b_obj, b)) return NULL;
    
    /* r = a @ b, store into self->m */
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            self->m[row*4 + col] = 
                a[row*4 + 0] * b[0*4 + col] +
                a[row*4 + 1] * b[1*4 + col] +
                a[row*4 + 2] * b[2*4 + col] +
                a[row*4 + 3] * b[3*4 + col];
        }
    }
    Py_RETURN_NONE;
}

/* Mat4.set_translation(x, y, z) - Set as translation matrix in-place */
static PyObject *Mat4_set_translation(Mat4Object *self, PyObject *args) {
    double x, y, z;
    if (!PyArg_ParseTuple(args, "ddd", &x, &y, &z)) return NULL;
    
    /* Identity with translation in row 3 */
    self->m[0] = 1.0; self->m[1] = 0.0; self->m[2] = 0.0; self->m[3] = 0.0;
    self->m[4] = 0.0; self->m[5] = 1.0; self->m[6] = 0.0; self->m[7] = 0.0;
    self->m[8] = 0.0; self->m[9] = 0.0; self->m[10] = 1.0; self->m[11] = 0.0;
    self->m[12] = x;  self->m[13] = y;  self->m[14] = z;  self->m[15] = 1.0;
    Py_RETURN_NONE;
}

/* Mat4.set_scale(sx, sy, sz) - Set as scale matrix in-place */
static PyObject *Mat4_set_scale(Mat4Object *self, PyObject *args) {
    double sx, sy, sz;
    if (!PyArg_ParseTuple(args, "ddd", &sx, &sy, &sz)) return NULL;
    
    for (int i = 0; i < 16; i++) self->m[i] = 0.0;
    self->m[0] = sx; self->m[5] = sy; self->m[10] = sz; self->m[15] = 1.0;
    Py_RETURN_NONE;
}

/* Mat4.set_rotation_xyz(rx, ry, rz) - Set as XYZ rotation matrix in-place (angles in radians) */
static PyObject *Mat4_set_rotation_xyz(Mat4Object *self, PyObject *args) {
    double rx, ry, rz;
    if (!PyArg_ParseTuple(args, "ddd", &rx, &ry, &rz)) return NULL;
    
    double cx = cos(rx), sx = sin(rx);
    double cy = cos(ry), sy = sin(ry);
    double cz = cos(rz), sz = sin(rz);
    
    /* Combined XYZ rotation matrix: Rx @ Ry @ Rz */
    self->m[0] = cy*cz;              self->m[1] = cy*sz;              self->m[2] = -sy;     self->m[3] = 0.0;
    self->m[4] = sx*sy*cz - cx*sz;   self->m[5] = sx*sy*sz + cx*cz;   self->m[6] = sx*cy;   self->m[7] = 0.0;
    self->m[8] = cx*sy*cz + sx*sz;   self->m[9] = cx*sy*sz - sx*cz;   self->m[10] = cx*cy;  self->m[11] = 0.0;
    self->m[12] = 0.0;               self->m[13] = 0.0;               self->m[14] = 0.0;    self->m[15] = 1.0;
    Py_RETURN_NONE;
}

/* Mat4.set_trs(tx, ty, tz, rx, ry, rz, sx, sy, sz) - Build full TRS transform in-place 
 * Rotation angles in DEGREES, transform order is SRT (scale, then rotate, then translate)
 */
static PyObject *Mat4_set_trs(Mat4Object *self, PyObject *args) {
    double tx, ty, tz, rdx, rdy, rdz, scx, scy, scz;
    if (!PyArg_ParseTuple(args, "ddddddddd", &tx, &ty, &tz, &rdx, &rdy, &rdz, &scx, &scy, &scz)) 
        return NULL;
    
    /* Convert to radians */
    double rx = rdx * M_PI / 180.0;
    double ry = rdy * M_PI / 180.0;
    double rz = rdz * M_PI / 180.0;
    
    double cx = cos(rx), snx = sin(rx);
    double cy = cos(ry), sny = sin(ry);
    double cz = cos(rz), snz = sin(rz);
    
    /* Build combined SRT matrix directly: T @ R @ S
     * For XYZ rotation order, the rotation part is: Rx @ Ry @ Rz
     * Then we multiply by scale and add translation
     */
    
    /* Rotation matrix (XYZ order) */
    double r00 = cy*cz,               r01 = cy*snz,              r02 = -sny;
    double r10 = snx*sny*cz - cx*snz, r11 = snx*sny*snz + cx*cz, r12 = snx*cy;
    double r20 = cx*sny*cz + snx*snz, r21 = cx*sny*snz - snx*cz, r22 = cx*cy;
    
    /* Apply scale to rotation columns */
    self->m[0] = r00 * scx;  self->m[1] = r01 * scx;  self->m[2] = r02 * scx;  self->m[3] = 0.0;
    self->m[4] = r10 * scy;  self->m[5] = r11 * scy;  self->m[6] = r12 * scy;  self->m[7] = 0.0;
    self->m[8] = r20 * scz;  self->m[9] = r21 * scz;  self->m[10] = r22 * scz; self->m[11] = 0.0;
    self->m[12] = tx;        self->m[13] = ty;        self->m[14] = tz;        self->m[15] = 1.0;
    
    Py_RETURN_NONE;
}

/* Mat4.invert_into() - Invert matrix in-place */
static PyObject *Mat4_invert_into(Mat4Object *self, PyObject *Py_UNUSED(ignored)) {
    double *m = self->m;
    double inv[16];
    
    /* Compute inverse using cofactor expansion */
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
    for (int i = 0; i < 16; i++) self->m[i] = inv[i] * inv_det;
    
    Py_RETURN_NONE;
}

/* Mat4.pre_multiply(other) - self = other @ self (in-place) */
static PyObject *Mat4_pre_multiply(Mat4Object *self, PyObject *args) {
    PyObject *other_obj;
    double other[16], result[16];
    
    if (!PyArg_ParseTuple(args, "O", &other_obj)) return NULL;
    if (!Mat4_extract(other_obj, other)) return NULL;
    
    /* result = other @ self */
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            result[row*4 + col] = 
                other[row*4 + 0] * self->m[0*4 + col] +
                other[row*4 + 1] * self->m[1*4 + col] +
                other[row*4 + 2] * self->m[2*4 + col] +
                other[row*4 + 3] * self->m[3*4 + col];
        }
    }
    
    for (int i = 0; i < 16; i++) self->m[i] = result[i];
    Py_RETURN_NONE;
}

/* Mat4.post_multiply(other) - self = self @ other (in-place) */
static PyObject *Mat4_post_multiply(Mat4Object *self, PyObject *args) {
    PyObject *other_obj;
    double other[16], result[16];
    
    if (!PyArg_ParseTuple(args, "O", &other_obj)) return NULL;
    if (!Mat4_extract(other_obj, other)) return NULL;
    
    /* result = self @ other */
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            result[row*4 + col] = 
                self->m[row*4 + 0] * other[0*4 + col] +
                self->m[row*4 + 1] * other[1*4 + col] +
                self->m[row*4 + 2] * other[2*4 + col] +
                self->m[row*4 + 3] * other[3*4 + col];
        }
    }
    
    for (int i = 0; i < 16; i++) self->m[i] = result[i];
    Py_RETURN_NONE;
}

/* Mat4 methods */
static PyMethodDef Mat4_methods[] = {
    {"transform_point", (PyCFunction)Mat4_transform_point, METH_VARARGS, 
     "Transform a point by this matrix, returns Vec3"},
    {"bilinear_transform", (PyCFunction)Mat4_bilinear_transform, METH_VARARGS,
     "Bilinear interpolate + transform: bilinear_transform(t, s, z_offset, p0, p1, p2, p3)"},
    {"transform_direction", (PyCFunction)Mat4_transform_direction, METH_VARARGS, 
     "Transform a direction (no translation), returns Vec3"},
    {"inverse", (PyCFunction)Mat4_inverse, METH_NOARGS, "Return inverted matrix"},
    {"to_tuple", (PyCFunction)Mat4_to_tuple, METH_NOARGS, "Convert to nested tuple"},
    {"batch_transform_points", (PyCFunction)Mat4_batch_transform_points, METH_VARARGS,
     "Transform multiple points at once"},
    {"batch_bilinear_transform", (PyCFunction)Mat4_batch_bilinear_transform, METH_VARARGS,
     "Bilinear interpolate + transform multiple points"},
    /* In-place methods */
    {"set_identity", (PyCFunction)Mat4_set_identity, METH_NOARGS, 
     "Reset to identity matrix in-place"},
    {"set_from", (PyCFunction)Mat4_set_from, METH_VARARGS, 
     "Copy values from another matrix in-place"},
    {"multiply_into", (PyCFunction)Mat4_multiply_into, METH_VARARGS, 
     "Compute a @ b and store result in self"},
    {"set_translation", (PyCFunction)Mat4_set_translation, METH_VARARGS, 
     "Set as translation matrix in-place"},
    {"set_scale", (PyCFunction)Mat4_set_scale, METH_VARARGS, 
     "Set as scale matrix in-place"},
    {"set_rotation_xyz", (PyCFunction)Mat4_set_rotation_xyz, METH_VARARGS, 
     "Set as XYZ rotation matrix in-place (angles in radians)"},
    {"set_trs", (PyCFunction)Mat4_set_trs, METH_VARARGS, 
     "Build TRS transform in-place (angles in degrees)"},
    {"invert_into", (PyCFunction)Mat4_invert_into, METH_NOARGS, 
     "Invert matrix in-place"},
    {"pre_multiply", (PyCFunction)Mat4_pre_multiply, METH_VARARGS, 
     "self = other @ self (in-place)"},
    {"post_multiply", (PyCFunction)Mat4_post_multiply, METH_VARARGS, 
     "self = self @ other (in-place)"},
    {NULL}
};

/* Mat4 type definition */
static PyTypeObject Mat4Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_vec3.Mat4",
    .tp_doc = "4x4 matrix with fast operations (contiguous C storage)",
    .tp_basicsize = sizeof(Mat4Object),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Mat4_new,
    .tp_init = (initproc)Mat4_init,
    .tp_repr = (reprfunc)Mat4_repr,
    .tp_as_number = &Mat4_as_number,
    .tp_as_sequence = &Mat4_as_sequence,
    .tp_methods = Mat4_methods,
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

/* mat4_multiply(a, b) - Multiply two 4x4 matrices, returns tuple of tuples */
static PyObject *vec3_mat4_multiply(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj;
    double a[4][4], b[4][4], r[4][4];
    
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    
    /* Extract matrices */
    for (int i = 0; i < 4; i++) {
        PyObject *row_a = PySequence_GetItem(a_obj, i);
        PyObject *row_b = PySequence_GetItem(b_obj, i);
        if (!row_a || !row_b) {
            Py_XDECREF(row_a);
            Py_XDECREF(row_b);
            return NULL;
        }
        for (int j = 0; j < 4; j++) {
            PyObject *item_a = PySequence_GetItem(row_a, j);
            PyObject *item_b = PySequence_GetItem(row_b, j);
            if (!item_a || !item_b) {
                Py_XDECREF(item_a);
                Py_XDECREF(item_b);
                Py_DECREF(row_a);
                Py_DECREF(row_b);
                return NULL;
            }
            a[i][j] = PyFloat_AsDouble(item_a);
            b[i][j] = PyFloat_AsDouble(item_b);
            Py_DECREF(item_a);
            Py_DECREF(item_b);
        }
        Py_DECREF(row_a);
        Py_DECREF(row_b);
    }
    
    /* Matrix multiply: r = a @ b */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            r[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j] + a[i][3]*b[3][j];
        }
    }
    
    return Py_BuildValue("((dddd)(dddd)(dddd)(dddd))",
        r[0][0], r[0][1], r[0][2], r[0][3],
        r[1][0], r[1][1], r[1][2], r[1][3],
        r[2][0], r[2][1], r[2][2], r[2][3],
        r[3][0], r[3][1], r[3][2], r[3][3]);
}

/* cross_product(a, b) - Cross product of two 3D vectors, returns tuple */
static PyObject *vec3_cross_product(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj;
    double ax, ay, az, bx, by, bz;
    
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    if (!Vec3_extract(a_obj, &ax, &ay, &az)) return NULL;
    if (!Vec3_extract(b_obj, &bx, &by, &bz)) return NULL;
    
    return Py_BuildValue("(ddd)", 
        ay*bz - az*by,
        az*bx - ax*bz,
        ax*by - ay*bx);
}

/* is_point_on_line_segment(point, p1, p2, tol) - Check if point is on line segment */
static PyObject *vec3_is_point_on_line_segment(PyObject *self, PyObject *args) {
    PyObject *point_obj, *p1_obj, *p2_obj;
    double px, py, pz, p1x, p1y, p1z, p2x, p2y, p2z;
    double tol = 1e-6;
    
    if (!PyArg_ParseTuple(args, "OOO|d", &point_obj, &p1_obj, &p2_obj, &tol)) return NULL;
    
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
    
    /* Cross product to check collinearity */
    double cx = sy*vz - sz*vy;
    double cy = sz*vx - sx*vz;
    double cz = sx*vy - sy*vx;
    double cross_mag = sqrt(cx*cx + cy*cy + cz*cz);
    
    if (cross_mag > tol) {
        Py_RETURN_FALSE;
    }
    
    /* Check if point is within segment bounds */
    double seg_len_sq = sx*sx + sy*sy + sz*sz;
    if (seg_len_sq < 1e-15) {
        /* Degenerate segment - check if point equals p1 */
        double d = (px-p1x)*(px-p1x) + (py-p1y)*(py-p1y) + (pz-p1z)*(pz-p1z);
        if (d < tol*tol) Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    }
    
    double t = (vx*sx + vy*sy + vz*sz) / seg_len_sq;
    if (t >= -tol && t <= 1.0 + tol) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/* Module method definitions */
/* bilinear_interpolate(t, s, p0, p1, p2, p3) - Bilinear interpolation on a quad
 * Returns the interpolated point as a tuple.
 * Points p0-p3 are corners: p0=start-bottom, p1=end-bottom, p2=end-top, p3=start-top
 */
static PyObject *vec3_bilinear_interpolate(PyObject *self, PyObject *args) {
    double t, s;
    PyObject *p0_obj, *p1_obj, *p2_obj, *p3_obj;
    double p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z;
    
    if (!PyArg_ParseTuple(args, "ddOOOO", &t, &s, &p0_obj, &p1_obj, &p2_obj, &p3_obj)) 
        return NULL;
    
    if (!Vec3_extract(p0_obj, &p0x, &p0y, &p0z)) return NULL;
    if (!Vec3_extract(p1_obj, &p1x, &p1y, &p1z)) return NULL;
    if (!Vec3_extract(p2_obj, &p2x, &p2y, &p2z)) return NULL;
    if (!Vec3_extract(p3_obj, &p3x, &p3y, &p3z)) return NULL;
    
    /* l1 = lerp(t, p0, p1) = p0 + t * (p1 - p0) */
    double l1x = p0x + t * (p1x - p0x);
    double l1y = p0y + t * (p1y - p0y);
    double l1z = p0z + t * (p1z - p0z);
    
    /* l2 = lerp(t, p3, p2) = p3 + t * (p2 - p3) */
    double l2x = p3x + t * (p2x - p3x);
    double l2y = p3y + t * (p2y - p3y);
    double l2z = p3z + t * (p2z - p3z);
    
    /* result = lerp(s, l1, l2) = l1 + s * (l2 - l1) */
    return Py_BuildValue("(ddd)",
        l1x + s * (l2x - l1x),
        l1y + s * (l2y - l1y),
        l1z + s * (l2z - l1z)
    );
}

/* project_point_on_ray(point, ray_origin, ray_direction, ray_length) 
 * Returns the t-value (0-1 range) of the projection.
 */
static PyObject *vec3_project_point_on_ray(PyObject *self, PyObject *args) {
    PyObject *point_obj, *origin_obj, *dir_obj;
    double ray_length;
    double px, py, pz, ox, oy, oz, dx, dy, dz;
    
    if (!PyArg_ParseTuple(args, "OOOd", &point_obj, &origin_obj, &dir_obj, &ray_length)) 
        return NULL;
    
    if (!Vec3_extract(point_obj, &px, &py, &pz)) return NULL;
    if (!Vec3_extract(origin_obj, &ox, &oy, &oz)) return NULL;
    if (!Vec3_extract(dir_obj, &dx, &dy, &dz)) return NULL;
    
    /* diff = point - origin */
    double diff_x = px - ox;
    double diff_y = py - oy;
    double diff_z = pz - oz;
    
    /* dot(diff, direction) / length */
    double dot = diff_x * dx + diff_y * dy + diff_z * dz;
    return PyFloat_FromDouble(dot / ray_length);
}

/* batch_transform_points(points, matrix) - Transform multiple points by a 4x4 matrix
 * Returns a list of tuples.
 */
static PyObject *vec3_batch_transform_points(PyObject *self, PyObject *args) {
    PyObject *points_obj, *matrix_obj;
    double m[4][4];
    
    if (!PyArg_ParseTuple(args, "OO", &points_obj, &matrix_obj)) return NULL;
    
    /* Extract 4x4 matrix */
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
    
    /* Process all points */
    Py_ssize_t n = PySequence_Size(points_obj);
    if (n < 0) return NULL;
    
    PyObject *result = PyList_New(n);
    if (!result) return NULL;
    
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *point = PySequence_GetItem(points_obj, i);
        if (!point) {
            Py_DECREF(result);
            return NULL;
        }
        
        double x, y, z;
        if (!Vec3_extract(point, &x, &y, &z)) {
            Py_DECREF(point);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(point);
        
        double w = x * m[0][3] + y * m[1][3] + z * m[2][3] + m[3][3];
        double inv_w = (fabs(w) < 1e-10) ? 0.0 : (1.0 / w);
        
        PyObject *transformed = Py_BuildValue("(ddd)",
            (x * m[0][0] + y * m[1][0] + z * m[2][0] + m[3][0]) * inv_w,
            (x * m[0][1] + y * m[1][1] + z * m[2][1] + m[3][1]) * inv_w,
            (x * m[0][2] + y * m[1][2] + z * m[2][2] + m[3][2]) * inv_w
        );
        
        if (!transformed) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, i, transformed);
    }
    
    return result;
}

/* batch_bilinear_transform(points_with_offsets, quad_points, matrix)
 * Perform bilinear interpolation + matrix transform in a single call for multiple points.
 * points_with_offsets: list of (t, s, z_offset) tuples
 * quad_points: (p0, p1, p2, p3) - the quad corners for bilinear interpolation  
 * matrix: 4x4 transformation matrix
 * Returns: list of tuples (x, y, z)
 */
static PyObject *vec3_batch_bilinear_transform(PyObject *self, PyObject *args) {
    PyObject *points_obj, *quad_obj, *matrix_obj;
    double m[4][4];
    double q0x, q0y, q0z, q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;
    
    if (!PyArg_ParseTuple(args, "OOO", &points_obj, &quad_obj, &matrix_obj)) return NULL;
    
    /* Extract quad points */
    PyObject *p0 = PySequence_GetItem(quad_obj, 0);
    PyObject *p1 = PySequence_GetItem(quad_obj, 1);
    PyObject *p2 = PySequence_GetItem(quad_obj, 2);
    PyObject *p3 = PySequence_GetItem(quad_obj, 3);
    if (!p0 || !p1 || !p2 || !p3) {
        Py_XDECREF(p0); Py_XDECREF(p1); Py_XDECREF(p2); Py_XDECREF(p3);
        return NULL;
    }
    if (!Vec3_extract(p0, &q0x, &q0y, &q0z) ||
        !Vec3_extract(p1, &q1x, &q1y, &q1z) ||
        !Vec3_extract(p2, &q2x, &q2y, &q2z) ||
        !Vec3_extract(p3, &q3x, &q3y, &q3z)) {
        Py_DECREF(p0); Py_DECREF(p1); Py_DECREF(p2); Py_DECREF(p3);
        return NULL;
    }
    Py_DECREF(p0); Py_DECREF(p1); Py_DECREF(p2); Py_DECREF(p3);
    
    /* Extract 4x4 matrix */
    for (int i = 0; i < 4; i++) {
        PyObject *row = PySequence_GetItem(matrix_obj, i);
        if (!row) return NULL;
        for (int j = 0; j < 4; j++) {
            PyObject *item = PySequence_GetItem(row, j);
            if (!item) { Py_DECREF(row); return NULL; }
            m[i][j] = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) { Py_DECREF(row); return NULL; }
        }
        Py_DECREF(row);
    }
    
    /* Process all points */
    Py_ssize_t n = PySequence_Size(points_obj);
    PyObject *result = PyList_New(n);
    if (!result) return NULL;
    
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *point_tuple = PySequence_GetItem(points_obj, i);
        if (!point_tuple) { Py_DECREF(result); return NULL; }
        
        double t, s, z_offset;
        if (!Vec3_extract(point_tuple, &t, &s, &z_offset)) {
            Py_DECREF(point_tuple);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(point_tuple);
        
        /* Bilinear interpolation: l1 = lerp(t, p0, p1), l2 = lerp(t, p3, p2), p = lerp(s, l1, l2) */
        double l1x = q0x + t * (q1x - q0x);
        double l1y = q0y + t * (q1y - q0y);
        double l1z = q0z + t * (q1z - q0z);
        
        double l2x = q3x + t * (q2x - q3x);
        double l2y = q3y + t * (q2y - q3y);
        double l2z = q3z + t * (q2z - q3z);
        
        double ix = l1x + s * (l2x - l1x);
        double iy = l1y + s * (l2y - l1y);
        double iz = l1z + s * (l2z - l1z) + z_offset;
        
        /* Matrix transform */
        double w = ix * m[0][3] + iy * m[1][3] + iz * m[2][3] + m[3][3];
        double inv_w = (fabs(w) < 1e-10) ? 0.0 : (1.0 / w);
        
        PyObject *transformed = Py_BuildValue("(ddd)",
            (ix * m[0][0] + iy * m[1][0] + iz * m[2][0] + m[3][0]) * inv_w,
            (ix * m[0][1] + iy * m[1][1] + iz * m[2][1] + m[3][1]) * inv_w,
            (ix * m[0][2] + iy * m[1][2] + iz * m[2][2] + m[3][2]) * inv_w
        );
        
        if (!transformed) { Py_DECREF(result); return NULL; }
        PyList_SET_ITEM(result, i, transformed);
    }
    
    return result;
}


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
    {"mat4_multiply", vec3_mat4_multiply, METH_VARARGS,
     "Multiply two 4x4 matrices, returns tuple of tuples"},
    {"cross_product", vec3_cross_product, METH_VARARGS,
     "Cross product of two 3D vectors, returns tuple"},
    {"is_point_on_line_segment", vec3_is_point_on_line_segment, METH_VARARGS,
     "Check if point lies on line segment"},
    {"bilinear_interpolate", vec3_bilinear_interpolate, METH_VARARGS,
     "Bilinear interpolation on a quad: bilinear_interpolate(t, s, p0, p1, p2, p3)"},
    {"project_point_on_ray", vec3_project_point_on_ray, METH_VARARGS,
     "Project a point onto a ray and return the t-value"},
    {"batch_transform_points", vec3_batch_transform_points, METH_VARARGS,
     "Transform multiple points by a 4x4 matrix at once"},
    {"batch_bilinear_transform", vec3_batch_bilinear_transform, METH_VARARGS,
     "Bilinear interpolate + transform multiple points in one call"},
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
    
    if (PyType_Ready(&Mat4Type) < 0)
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
    
    Py_INCREF(&Mat4Type);
    if (PyModule_AddObject(m, "Mat4", (PyObject *)&Mat4Type) < 0) {
        Py_DECREF(&Mat4Type);
        Py_DECREF(&Vec3Type);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}
