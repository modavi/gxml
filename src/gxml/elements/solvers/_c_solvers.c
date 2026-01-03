/*
 * _c_solvers.c - High-performance panel solver C extension for GXML
 * 
 * This module provides the complete solver pipeline in C:
 *   1. batch_find_intersections() - Find all centerline intersections
 *   2. batch_solve_faces() - Compute face segmentation and trim bounds
 *   3. batch_build_geometry() - Generate vertices and indices
 * 
 * Data stays in C memory between stages to avoid Python marshaling overhead.
 * 
 * Build with: python setup_c_solvers.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define ENDPOINT_THRESHOLD 0.01
#define TOLERANCE 1e-6
#define MAX_PANELS_PER_INTERSECTION 16
#define MAX_INTERSECTIONS_PER_PANEL 64

/* Intersection types */
#define INTERSECTION_JOINT 1
#define INTERSECTION_T_JUNCTION 2
#define INTERSECTION_CROSSING 3

/* Face sides */
#define FACE_FRONT 0
#define FACE_BACK 1
#define FACE_TOP 2
#define FACE_BOTTOM 3
#define FACE_START 4
#define FACE_END 5
#define NUM_FACES 6

/* ============================================================================
 * Data Structures
 * ============================================================================ */

typedef struct {
    double start[3];
    double end[3];
    double width;
    double height;
    double thickness;
    double transform[16];
} PanelData;

typedef struct {
    int panel_idx;
    double t;
} IntersectionEntry;

typedef struct {
    int type;
    double position[3];
    int num_panels;
    IntersectionEntry panels[MAX_PANELS_PER_INTERSECTION];
} IntersectionData;

typedef struct {
    double corners[4][2];
} FaceSegment;

typedef struct {
    int panel_idx;
    int num_segments[NUM_FACES];
    FaceSegment* segments[NUM_FACES];
} SegmentedPanelData;

typedef struct {
    int num_panels;
    PanelData* panels;
    int num_intersections;
    IntersectionData* intersections;
    int* panel_intersection_counts;
    int** panel_intersection_indices;
    SegmentedPanelData* segmented_panels;
    int num_vertices;
    int num_indices;
    double* vertices;
    int* indices;
} SolverContext;

/* ============================================================================
 * Vector Math
 * ============================================================================ */

static inline void vec3_sub(const double* a, const double* b, double* out) {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
}

static inline double vec3_dot(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void vec3_cross(const double* a, const double* b, double* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline double vec3_length_sq(const double* v) {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

static inline void vec3_lerp(const double* a, const double* b, double t, double* out) {
    out[0] = a[0] + t * (b[0] - a[0]);
    out[1] = a[1] + t * (b[1] - a[1]);
    out[2] = a[2] + t * (b[2] - a[2]);
}

static inline void transform_point(const double* m, const double* p, double* out) {
    out[0] = m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3];
    out[1] = m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7];
    out[2] = m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11];
}

static inline void bilinear_interp(double t, double s,
                                    const double* p0, const double* p1,
                                    const double* p2, const double* p3,
                                    double* out) {
    double bottom[3], top[3];
    vec3_lerp(p0, p1, t, bottom);
    vec3_lerp(p3, p2, t, top);
    vec3_lerp(bottom, top, s, out);
}

/* ============================================================================
 * Segment Intersection
 * ============================================================================ */

static int segment_intersection(
    const double* p1, const double* p2,
    const double* p3, const double* p4,
    double tol, double* t1_out, double* t2_out, double* pos
) {
    double d1[3], d2[3], w[3];
    vec3_sub(p2, p1, d1);
    vec3_sub(p4, p3, d2);
    vec3_sub(p3, p1, w);
    
    double cross[3];
    vec3_cross(d1, d2, cross);
    double denom = vec3_length_sq(cross);
    
    double tol_sq = tol * tol;
    if (denom < tol_sq) return 0;
    
    double wcd2[3], wcd1[3];
    vec3_cross(w, d2, wcd2);
    double t1 = vec3_dot(wcd2, cross) / denom;
    if (t1 < -tol || t1 > 1.0 + tol) return 0;
    
    vec3_cross(w, d1, wcd1);
    double t2 = vec3_dot(wcd1, cross) / denom;
    if (t2 < -tol || t2 > 1.0 + tol) return 0;
    
    double i1[3], i2[3], diff[3];
    vec3_lerp(p1, p2, t1, i1);
    vec3_lerp(p3, p4, t2, i2);
    vec3_sub(i1, i2, diff);
    if (vec3_length_sq(diff) >= tol_sq) return 0;
    
    *t1_out = t1; *t2_out = t2;
    pos[0] = i1[0]; pos[1] = i1[1]; pos[2] = i1[2];
    return 1;
}

/* ============================================================================
 * Context Management
 * ============================================================================ */

static SolverContext* context_create(int num_panels) {
    SolverContext* ctx = (SolverContext*)calloc(1, sizeof(SolverContext));
    if (!ctx) return NULL;
    
    ctx->num_panels = num_panels;
    ctx->panels = (PanelData*)calloc(num_panels, sizeof(PanelData));
    
    int max_inter = num_panels * (num_panels - 1) / 2;
    ctx->intersections = (IntersectionData*)calloc(max_inter, sizeof(IntersectionData));
    
    ctx->panel_intersection_counts = (int*)calloc(num_panels, sizeof(int));
    ctx->panel_intersection_indices = (int**)calloc(num_panels, sizeof(int*));
    for (int i = 0; i < num_panels; i++) {
        ctx->panel_intersection_indices[i] = (int*)calloc(MAX_INTERSECTIONS_PER_PANEL, sizeof(int));
    }
    
    ctx->segmented_panels = (SegmentedPanelData*)calloc(num_panels, sizeof(SegmentedPanelData));
    for (int i = 0; i < num_panels; i++) {
        ctx->segmented_panels[i].panel_idx = i;
    }
    
    return ctx;
}

static void context_free(SolverContext* ctx) {
    if (!ctx) return;
    free(ctx->panels);
    free(ctx->intersections);
    free(ctx->panel_intersection_counts);
    for (int i = 0; i < ctx->num_panels; i++) {
        free(ctx->panel_intersection_indices[i]);
        for (int f = 0; f < NUM_FACES; f++) {
            free(ctx->segmented_panels[i].segments[f]);
        }
    }
    free(ctx->panel_intersection_indices);
    free(ctx->segmented_panels);
    free(ctx->vertices);
    free(ctx->indices);
    free(ctx);
}

static void context_destructor(PyObject* capsule) {
    SolverContext* ctx = (SolverContext*)PyCapsule_GetPointer(capsule, "SolverContext");
    context_free(ctx);
}

/* ============================================================================
 * Stage 1: Find Intersections
 * ============================================================================ */

static void solve_intersections(SolverContext* ctx) {
    int n = ctx->num_panels;
    ctx->num_intersections = 0;
    memset(ctx->panel_intersection_counts, 0, n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        PanelData* p1 = &ctx->panels[i];
        for (int j = i + 1; j < n; j++) {
            PanelData* p2 = &ctx->panels[j];
            double t1, t2, pos[3];
            if (segment_intersection(p1->start, p1->end, p2->start, p2->end, TOLERANCE, &t1, &t2, pos)) {
                int at_ep1 = (t1 < ENDPOINT_THRESHOLD || t1 > 1.0 - ENDPOINT_THRESHOLD);
                int at_ep2 = (t2 < ENDPOINT_THRESHOLD || t2 > 1.0 - ENDPOINT_THRESHOLD);
                
                int type = (at_ep1 && at_ep2) ? INTERSECTION_JOINT :
                           (at_ep1 || at_ep2) ? INTERSECTION_T_JUNCTION : INTERSECTION_CROSSING;
                
                IntersectionData* inter = &ctx->intersections[ctx->num_intersections];
                inter->type = type;
                inter->position[0] = pos[0]; inter->position[1] = pos[1]; inter->position[2] = pos[2];
                inter->num_panels = 2;
                inter->panels[0].panel_idx = i; inter->panels[0].t = t1;
                inter->panels[1].panel_idx = j; inter->panels[1].t = t2;
                
                int idx = ctx->num_intersections;
                if (ctx->panel_intersection_counts[i] < MAX_INTERSECTIONS_PER_PANEL)
                    ctx->panel_intersection_indices[i][ctx->panel_intersection_counts[i]++] = idx;
                if (ctx->panel_intersection_counts[j] < MAX_INTERSECTIONS_PER_PANEL)
                    ctx->panel_intersection_indices[j][ctx->panel_intersection_counts[j]++] = idx;
                
                ctx->num_intersections++;
            }
        }
    }
}

/* ============================================================================
 * Stage 2: Solve Faces
 * ============================================================================ */

static void allocate_segments(SegmentedPanelData* sp, int face, int count) {
    sp->num_segments[face] = count;
    sp->segments[face] = (FaceSegment*)calloc(count, sizeof(FaceSegment));
}

static void set_corners(FaceSegment* seg, double t0, double s0, double t1, double s1,
                        double t2, double s2, double t3, double s3) {
    seg->corners[0][0] = t0; seg->corners[0][1] = s0;
    seg->corners[1][0] = t1; seg->corners[1][1] = s1;
    seg->corners[2][0] = t2; seg->corners[2][1] = s2;
    seg->corners[3][0] = t3; seg->corners[3][1] = s3;
}

static void solve_faces_for_panel(SolverContext* ctx, int panel_idx) {
    SegmentedPanelData* sp = &ctx->segmented_panels[panel_idx];
    PanelData* panel = &ctx->panels[panel_idx];
    
    double t_splits[MAX_INTERSECTIONS_PER_PANEL + 2];
    int num_splits = 0;
    t_splits[num_splits++] = 0.0;
    
    int has_start = 0, has_end = 0;
    
    for (int i = 0; i < ctx->panel_intersection_counts[panel_idx]; i++) {
        int idx = ctx->panel_intersection_indices[panel_idx][i];
        IntersectionData* inter = &ctx->intersections[idx];
        double t = -1;
        for (int p = 0; p < inter->num_panels; p++) {
            if (inter->panels[p].panel_idx == panel_idx) { t = inter->panels[p].t; break; }
        }
        if (t < 0) continue;
        if (t < ENDPOINT_THRESHOLD) has_start = 1;
        else if (t > 1.0 - ENDPOINT_THRESHOLD) has_end = 1;
        else t_splits[num_splits++] = t;
    }
    t_splits[num_splits++] = 1.0;
    
    /* Sort */
    for (int i = 0; i < num_splits - 1; i++)
        for (int j = i + 1; j < num_splits; j++)
            if (t_splits[j] < t_splits[i]) { double tmp = t_splits[i]; t_splits[i] = t_splits[j]; t_splits[j] = tmp; }
    
    /* Dedupe */
    int unique = 1;
    for (int i = 1; i < num_splits; i++)
        if (t_splits[i] - t_splits[unique-1] > TOLERANCE) t_splits[unique++] = t_splits[i];
    num_splits = unique;
    
    int n_segs = num_splits - 1;
    
    /* FRONT/BACK/TOP/BOTTOM */
    for (int face = FACE_FRONT; face <= FACE_BOTTOM; face++) {
        allocate_segments(sp, face, n_segs);
        for (int s = 0; s < n_segs; s++)
            set_corners(&sp->segments[face][s], t_splits[s], 0.0, t_splits[s+1], 0.0, t_splits[s+1], 1.0, t_splits[s], 1.0);
    }
    
    /* START */
    if (!has_start && panel->thickness > TOLERANCE) {
        allocate_segments(sp, FACE_START, 1);
        set_corners(&sp->segments[FACE_START][0], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0);
    } else sp->num_segments[FACE_START] = 0;
    
    /* END */
    if (!has_end && panel->thickness > TOLERANCE) {
        allocate_segments(sp, FACE_END, 1);
        set_corners(&sp->segments[FACE_END][0], 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0);
    } else sp->num_segments[FACE_END] = 0;
}

static void solve_faces(SolverContext* ctx) {
    for (int i = 0; i < ctx->num_panels; i++) solve_faces_for_panel(ctx, i);
}

/* ============================================================================
 * Stage 3: Build Geometry
 * ============================================================================ */

static void get_face_point(const PanelData* panel, int face, double t, double s, double* out) {
    double p0[3], p1[3], p2[3], p3[3];
    double ht = panel->thickness / 2.0;
    
    switch (face) {
        case FACE_FRONT:
            p0[0]=0; p0[1]=0; p0[2]=ht; p1[0]=1; p1[1]=0; p1[2]=ht;
            p2[0]=1; p2[1]=1; p2[2]=ht; p3[0]=0; p3[1]=1; p3[2]=ht; break;
        case FACE_BACK:
            p0[0]=0; p0[1]=0; p0[2]=-ht; p1[0]=1; p1[1]=0; p1[2]=-ht;
            p2[0]=1; p2[1]=1; p2[2]=-ht; p3[0]=0; p3[1]=1; p3[2]=-ht; break;
        case FACE_TOP:
            p0[0]=0; p0[1]=1; p0[2]=-ht; p1[0]=1; p1[1]=1; p1[2]=-ht;
            p2[0]=1; p2[1]=1; p2[2]=ht; p3[0]=0; p3[1]=1; p3[2]=ht; break;
        case FACE_BOTTOM:
            p0[0]=0; p0[1]=0; p0[2]=ht; p1[0]=1; p1[1]=0; p1[2]=ht;
            p2[0]=1; p2[1]=0; p2[2]=-ht; p3[0]=0; p3[1]=0; p3[2]=-ht; break;
        case FACE_START:
            p0[0]=0; p0[1]=0; p0[2]=-ht; p1[0]=0; p1[1]=0; p1[2]=ht;
            p2[0]=0; p2[1]=1; p2[2]=ht; p3[0]=0; p3[1]=1; p3[2]=-ht; break;
        case FACE_END:
            p0[0]=1; p0[1]=0; p0[2]=ht; p1[0]=1; p1[1]=0; p1[2]=-ht;
            p2[0]=1; p2[1]=1; p2[2]=-ht; p3[0]=1; p3[1]=1; p3[2]=ht; break;
        default: out[0]=out[1]=out[2]=0; return;
    }
    
    double local[3];
    bilinear_interp(t, s, p0, p1, p2, p3, local);
    transform_point(panel->transform, local, out);
}

static void count_geometry(SolverContext* ctx, int* nv, int* ni) {
    *nv = 0; *ni = 0;
    for (int p = 0; p < ctx->num_panels; p++) {
        SegmentedPanelData* sp = &ctx->segmented_panels[p];
        for (int f = 0; f < NUM_FACES; f++) {
            *nv += sp->num_segments[f] * 4;
            *ni += sp->num_segments[f] * 6;
        }
    }
}

static void build_geometry(SolverContext* ctx) {
    int nv, ni;
    count_geometry(ctx, &nv, &ni);
    
    ctx->num_vertices = nv;
    ctx->num_indices = ni;
    ctx->vertices = (double*)malloc(nv * 3 * sizeof(double));
    ctx->indices = (int*)malloc(ni * sizeof(int));
    
    int vo = 0, io = 0;
    
    for (int p = 0; p < ctx->num_panels; p++) {
        PanelData* panel = &ctx->panels[p];
        SegmentedPanelData* sp = &ctx->segmented_panels[p];
        
        for (int f = 0; f < NUM_FACES; f++) {
            for (int s = 0; s < sp->num_segments[f]; s++) {
                FaceSegment* seg = &sp->segments[f][s];
                int base = vo / 3;
                
                for (int c = 0; c < 4; c++) {
                    get_face_point(panel, f, seg->corners[c][0], seg->corners[c][1], &ctx->vertices[vo]);
                    vo += 3;
                }
                
                ctx->indices[io++] = base + 0;
                ctx->indices[io++] = base + 1;
                ctx->indices[io++] = base + 2;
                ctx->indices[io++] = base + 0;
                ctx->indices[io++] = base + 2;
                ctx->indices[io++] = base + 3;
            }
        }
    }
}

/* ============================================================================
 * Python API
 * ============================================================================ */

static PyObject* py_create_context(PyObject* self, PyObject* args) {
    PyArrayObject *starts_arr, *ends_arr;
    PyObject *thick_obj = Py_None, *height_obj = Py_None;
    
    if (!PyArg_ParseTuple(args, "O!O!|OO", &PyArray_Type, &starts_arr, &PyArray_Type, &ends_arr, &thick_obj, &height_obj))
        return NULL;
    
    npy_intp n = PyArray_DIM(starts_arr, 0);
    SolverContext* ctx = context_create((int)n);
    if (!ctx) { PyErr_NoMemory(); return NULL; }
    
    PyArrayObject* starts = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)starts_arr, NPY_DOUBLE, 2, 2);
    PyArrayObject* ends = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)ends_arr, NPY_DOUBLE, 2, 2);
    if (!starts || !ends) { Py_XDECREF(starts); Py_XDECREF(ends); context_free(ctx); return NULL; }
    
    double* ps = (double*)PyArray_DATA(starts);
    double* pe = (double*)PyArray_DATA(ends);
    
    for (npy_intp i = 0; i < n; i++) {
        PanelData* p = &ctx->panels[i];
        p->start[0] = ps[i*3]; p->start[1] = ps[i*3+1]; p->start[2] = ps[i*3+2];
        p->end[0] = pe[i*3]; p->end[1] = pe[i*3+1]; p->end[2] = pe[i*3+2];
        
        double dx = p->end[0]-p->start[0], dy = p->end[1]-p->start[1], dz = p->end[2]-p->start[2];
        p->width = sqrt(dx*dx + dy*dy + dz*dz);
        p->height = 1.0;
        p->thickness = 0.1;
        
        memset(p->transform, 0, 16 * sizeof(double));
        p->transform[15] = 1.0;
        
        if (p->width > TOLERANCE) {
            double angle = atan2(-dz, dx);
            p->transform[0] = cos(angle) * p->width;
            p->transform[2] = -sin(angle) * p->width;
            p->transform[5] = p->height;
            p->transform[8] = sin(angle);
            p->transform[10] = cos(angle);
        } else { p->transform[0] = p->transform[5] = p->transform[10] = 1.0; }
        
        p->transform[3] = p->start[0]; p->transform[7] = p->start[1]; p->transform[11] = p->start[2];
    }
    
    Py_DECREF(starts); Py_DECREF(ends);
    
    if (thick_obj != Py_None && PyArray_Check(thick_obj)) {
        PyArrayObject* ta = (PyArrayObject*)PyArray_ContiguousFromAny(thick_obj, NPY_DOUBLE, 1, 1);
        if (ta) { double* pt = (double*)PyArray_DATA(ta); for (npy_intp i = 0; i < n; i++) ctx->panels[i].thickness = pt[i]; Py_DECREF(ta); }
    }
    if (height_obj != Py_None && PyArray_Check(height_obj)) {
        PyArrayObject* ha = (PyArrayObject*)PyArray_ContiguousFromAny(height_obj, NPY_DOUBLE, 1, 1);
        if (ha) { double* ph = (double*)PyArray_DATA(ha); for (npy_intp i = 0; i < n; i++) { ctx->panels[i].height = ph[i]; ctx->panels[i].transform[5] = ph[i]; } Py_DECREF(ha); }
    }
    
    return PyCapsule_New(ctx, "SolverContext", context_destructor);
}

static PyObject* py_solve_intersections(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    SolverContext* ctx = (SolverContext*)PyCapsule_GetPointer(cap, "SolverContext");
    if (!ctx) return NULL;
    solve_intersections(ctx);
    Py_RETURN_NONE;
}

static PyObject* py_solve_faces(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    SolverContext* ctx = (SolverContext*)PyCapsule_GetPointer(cap, "SolverContext");
    if (!ctx) return NULL;
    solve_faces(ctx);
    Py_RETURN_NONE;
}

static PyObject* py_build_geometry(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    SolverContext* ctx = (SolverContext*)PyCapsule_GetPointer(cap, "SolverContext");
    if (!ctx) return NULL;
    
    build_geometry(ctx);
    
    npy_intp vd[2] = {ctx->num_vertices, 3};
    npy_intp id[1] = {ctx->num_indices};
    PyArrayObject* verts = (PyArrayObject*)PyArray_SimpleNew(2, vd, NPY_DOUBLE);
    PyArrayObject* inds = (PyArrayObject*)PyArray_SimpleNew(1, id, NPY_INT32);
    if (!verts || !inds) { Py_XDECREF(verts); Py_XDECREF(inds); PyErr_NoMemory(); return NULL; }
    
    memcpy(PyArray_DATA(verts), ctx->vertices, ctx->num_vertices * 3 * sizeof(double));
    memcpy(PyArray_DATA(inds), ctx->indices, ctx->num_indices * sizeof(int));
    
    return Py_BuildValue("(NN)", verts, inds);
}

static PyObject* py_get_intersections(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    SolverContext* ctx = (SolverContext*)PyCapsule_GetPointer(cap, "SolverContext");
    if (!ctx) return NULL;
    
    int n = ctx->num_intersections;
    npy_intp d1[1] = {n}, d2[2] = {n, 3};
    
    PyArrayObject* types = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_INT32);
    PyArrayObject* pos = (PyArrayObject*)PyArray_SimpleNew(2, d2, NPY_DOUBLE);
    PyArrayObject* pi = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_INT32);
    PyArrayObject* pj = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_INT32);
    PyArrayObject* ti = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_DOUBLE);
    PyArrayObject* tj = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_DOUBLE);
    
    if (!types || !pos || !pi || !pj || !ti || !tj) {
        Py_XDECREF(types); Py_XDECREF(pos); Py_XDECREF(pi); Py_XDECREF(pj); Py_XDECREF(ti); Py_XDECREF(tj);
        PyErr_NoMemory(); return NULL;
    }
    
    int* pt = (int*)PyArray_DATA(types);
    double* pp = (double*)PyArray_DATA(pos);
    int* ppi = (int*)PyArray_DATA(pi);
    int* ppj = (int*)PyArray_DATA(pj);
    double* pti = (double*)PyArray_DATA(ti);
    double* ptj = (double*)PyArray_DATA(tj);
    
    for (int k = 0; k < n; k++) {
        IntersectionData* inter = &ctx->intersections[k];
        pt[k] = inter->type;
        pp[k*3] = inter->position[0]; pp[k*3+1] = inter->position[1]; pp[k*3+2] = inter->position[2];
        ppi[k] = inter->panels[0].panel_idx; ppj[k] = inter->panels[1].panel_idx;
        pti[k] = inter->panels[0].t; ptj[k] = inter->panels[1].t;
    }
    
    return Py_BuildValue("(NNNNNN)", types, pos, pi, pj, ti, tj);
}

static PyObject* py_solve_all(PyObject* self, PyObject* args) {
    PyArrayObject *starts_arr, *ends_arr;
    PyObject *thick_obj = Py_None, *height_obj = Py_None;
    
    if (!PyArg_ParseTuple(args, "O!O!|OO", &PyArray_Type, &starts_arr, &PyArray_Type, &ends_arr, &thick_obj, &height_obj))
        return NULL;
    
    npy_intp n = PyArray_DIM(starts_arr, 0);
    SolverContext* ctx = context_create((int)n);
    if (!ctx) { PyErr_NoMemory(); return NULL; }
    
    PyArrayObject* starts = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)starts_arr, NPY_DOUBLE, 2, 2);
    PyArrayObject* ends = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)ends_arr, NPY_DOUBLE, 2, 2);
    if (!starts || !ends) { Py_XDECREF(starts); Py_XDECREF(ends); context_free(ctx); return NULL; }
    
    double* ps = (double*)PyArray_DATA(starts);
    double* pe = (double*)PyArray_DATA(ends);
    
    for (npy_intp i = 0; i < n; i++) {
        PanelData* p = &ctx->panels[i];
        p->start[0] = ps[i*3]; p->start[1] = ps[i*3+1]; p->start[2] = ps[i*3+2];
        p->end[0] = pe[i*3]; p->end[1] = pe[i*3+1]; p->end[2] = pe[i*3+2];
        
        double dx = p->end[0]-p->start[0], dy = p->end[1]-p->start[1], dz = p->end[2]-p->start[2];
        p->width = sqrt(dx*dx + dy*dy + dz*dz);
        p->height = 1.0;
        p->thickness = 0.1;
        
        memset(p->transform, 0, 16 * sizeof(double));
        p->transform[15] = 1.0;
        
        if (p->width > TOLERANCE) {
            double angle = atan2(-dz, dx);
            p->transform[0] = cos(angle) * p->width;
            p->transform[2] = -sin(angle) * p->width;
            p->transform[5] = p->height;
            p->transform[8] = sin(angle);
            p->transform[10] = cos(angle);
        } else { p->transform[0] = p->transform[5] = p->transform[10] = 1.0; }
        
        p->transform[3] = p->start[0]; p->transform[7] = p->start[1]; p->transform[11] = p->start[2];
    }
    
    Py_DECREF(starts); Py_DECREF(ends);
    
    if (thick_obj != Py_None && PyArray_Check(thick_obj)) {
        PyArrayObject* ta = (PyArrayObject*)PyArray_ContiguousFromAny(thick_obj, NPY_DOUBLE, 1, 1);
        if (ta) { double* pt = (double*)PyArray_DATA(ta); for (npy_intp i = 0; i < n; i++) ctx->panels[i].thickness = pt[i]; Py_DECREF(ta); }
    }
    if (height_obj != Py_None && PyArray_Check(height_obj)) {
        PyArrayObject* ha = (PyArrayObject*)PyArray_ContiguousFromAny(height_obj, NPY_DOUBLE, 1, 1);
        if (ha) { double* ph = (double*)PyArray_DATA(ha); for (npy_intp i = 0; i < n; i++) { ctx->panels[i].height = ph[i]; ctx->panels[i].transform[5] = ph[i]; } Py_DECREF(ha); }
    }
    
    solve_intersections(ctx);
    solve_faces(ctx);
    build_geometry(ctx);
    
    npy_intp vd[2] = {ctx->num_vertices, 3};
    npy_intp id[1] = {ctx->num_indices};
    PyArrayObject* verts = (PyArrayObject*)PyArray_SimpleNew(2, vd, NPY_DOUBLE);
    PyArrayObject* inds = (PyArrayObject*)PyArray_SimpleNew(1, id, NPY_INT32);
    if (!verts || !inds) { Py_XDECREF(verts); Py_XDECREF(inds); context_free(ctx); PyErr_NoMemory(); return NULL; }
    
    memcpy(PyArray_DATA(verts), ctx->vertices, ctx->num_vertices * 3 * sizeof(double));
    memcpy(PyArray_DATA(inds), ctx->indices, ctx->num_indices * sizeof(int));
    
    int ni = ctx->num_intersections;
    context_free(ctx);
    
    return Py_BuildValue("(NNi)", verts, inds, ni);
}

static PyObject* batch_find_intersections(PyObject* self, PyObject* args) {
    PyArrayObject *starts_arr, *ends_arr;
    double tol = 1e-6;
    if (!PyArg_ParseTuple(args, "O!O!|d", &PyArray_Type, &starts_arr, &PyArray_Type, &ends_arr, &tol)) return NULL;
    
    npy_intp n = PyArray_DIM(starts_arr, 0);
    PyArrayObject* starts = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)starts_arr, NPY_DOUBLE, 2, 2);
    PyArrayObject* ends = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)ends_arr, NPY_DOUBLE, 2, 2);
    if (!starts || !ends) { Py_XDECREF(starts); Py_XDECREF(ends); return NULL; }
    
    double* ps = (double*)PyArray_DATA(starts);
    double* pe = (double*)PyArray_DATA(ends);
    
    npy_intp max = n * (n - 1) / 2;
    npy_intp* ri = (npy_intp*)malloc(max * sizeof(npy_intp));
    npy_intp* rj = (npy_intp*)malloc(max * sizeof(npy_intp));
    double* rt1 = (double*)malloc(max * sizeof(double));
    double* rt2 = (double*)malloc(max * sizeof(double));
    double* rp = (double*)malloc(max * 3 * sizeof(double));
    
    if (!ri || !rj || !rt1 || !rt2 || !rp) {
        free(ri); free(rj); free(rt1); free(rt2); free(rp);
        Py_DECREF(starts); Py_DECREF(ends); PyErr_NoMemory(); return NULL;
    }
    
    npy_intp cnt = 0;
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = i + 1; j < n; j++) {
            double t1, t2, pos[3];
            if (segment_intersection(&ps[i*3], &pe[i*3], &ps[j*3], &pe[j*3], tol, &t1, &t2, pos)) {
                ri[cnt] = i; rj[cnt] = j; rt1[cnt] = t1; rt2[cnt] = t2;
                rp[cnt*3] = pos[0]; rp[cnt*3+1] = pos[1]; rp[cnt*3+2] = pos[2];
                cnt++;
            }
        }
    }
    
    Py_DECREF(starts); Py_DECREF(ends);
    
    npy_intp d1[1] = {cnt}, d2[2] = {cnt, 3};
    PyArrayObject *oi = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_INTP);
    PyArrayObject *oj = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_INTP);
    PyArrayObject *ot1 = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_DOUBLE);
    PyArrayObject *ot2 = (PyArrayObject*)PyArray_SimpleNew(1, d1, NPY_DOUBLE);
    PyArrayObject *op = (PyArrayObject*)PyArray_SimpleNew(2, d2, NPY_DOUBLE);
    
    if (!oi || !oj || !ot1 || !ot2 || !op) {
        free(ri); free(rj); free(rt1); free(rt2); free(rp);
        Py_XDECREF(oi); Py_XDECREF(oj); Py_XDECREF(ot1); Py_XDECREF(ot2); Py_XDECREF(op);
        PyErr_NoMemory(); return NULL;
    }
    
    memcpy(PyArray_DATA(oi), ri, cnt * sizeof(npy_intp));
    memcpy(PyArray_DATA(oj), rj, cnt * sizeof(npy_intp));
    memcpy(PyArray_DATA(ot1), rt1, cnt * sizeof(double));
    memcpy(PyArray_DATA(ot2), rt2, cnt * sizeof(double));
    memcpy(PyArray_DATA(op), rp, cnt * 3 * sizeof(double));
    
    free(ri); free(rj); free(rt1); free(rt2); free(rp);
    return Py_BuildValue("(NNNNN)", oi, oj, ot1, ot2, op);
}

/* ============================================================================
 * Module Definition
 * ============================================================================ */

static PyMethodDef methods[] = {
    {"create_context", py_create_context, METH_VARARGS, "Create solver context from panel endpoints"},
    {"solve_intersections", py_solve_intersections, METH_VARARGS, "Find all intersections"},
    {"solve_faces", py_solve_faces, METH_VARARGS, "Compute face segmentation"},
    {"build_geometry", py_build_geometry, METH_VARARGS, "Generate mesh geometry"},
    {"get_intersections", py_get_intersections, METH_VARARGS, "Get intersection data"},
    {"solve_all", py_solve_all, METH_VARARGS, "Run complete pipeline"},
    {"batch_find_intersections", batch_find_intersections, METH_VARARGS, "Find intersections (standalone)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_c_solvers",
    "High-performance panel solver for GXML", -1, methods
};

PyMODINIT_FUNC PyInit__c_solvers(void) {
    import_array();
    return PyModule_Create(&module);
}
