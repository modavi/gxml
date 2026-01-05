/*
 * wasm_solvers.c - WebAssembly-compatible GXML solver
 * 
 * Standalone C implementation without Python/NumPy dependencies.
 * Designed for compilation with Emscripten to run in browsers.
 * 
 * Build: emcc wasm_solvers.c -O3 -o gxml_solvers.js -s WASM=1 \
 *        -s EXPORTED_FUNCTIONS="['_solve_intersections','_build_geometry','_malloc','_free']" \
 *        -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap']"
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

/* ============================================================================
 * Constants
 * ============================================================================ */

#define TOLERANCE 1e-6f
#define MAX_INTERSECTIONS 10000
#define MAX_PANELS 1000

/* ============================================================================
 * Data Structures (packed for JS interop)
 * ============================================================================ */

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    int panel_i;
    int panel_j;
    float t_i;
    float t_j;
    float pos_x, pos_y, pos_z;
} IntersectionResult;

/* Global buffers for results (avoids complex memory management in JS) */
static IntersectionResult g_intersections[MAX_INTERSECTIONS];
static int g_num_intersections = 0;

static float g_vertices[MAX_PANELS * 24 * 3];  /* Up to 24 verts per panel */
static int g_indices[MAX_PANELS * 36];          /* Up to 36 indices per panel */
static int g_num_vertices = 0;
static int g_num_indices = 0;

/* ============================================================================
 * Vector Math (inline for performance)
 * ============================================================================ */

static inline void vec3_sub(const float* a, const float* b, float* out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

static inline float vec3_dot(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline void vec3_cross(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline float vec3_length_sq(const float* v) {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

static inline void vec3_lerp(const float* a, const float* b, float t, float* out) {
    out[0] = a[0] + t * (b[0] - a[0]);
    out[1] = a[1] + t * (b[1] - a[1]);
    out[2] = a[2] + t * (b[2] - a[2]);
}

/* ============================================================================
 * Segment Intersection
 * ============================================================================ */

static int segment_intersect(
    const float* p1, const float* p2,
    const float* p3, const float* p4,
    float tol,
    float* t1_out, float* t2_out, float* pos_out
) {
    float d1[3], d2[3], w[3];
    vec3_sub(p2, p1, d1);
    vec3_sub(p4, p3, d2);
    vec3_sub(p3, p1, w);
    
    float cross[3];
    vec3_cross(d1, d2, cross);
    float denom = vec3_length_sq(cross);
    
    float tol_sq = tol * tol;
    if (denom < tol_sq) return 0;  /* Parallel lines */
    
    float wcd2[3], wcd1[3];
    vec3_cross(w, d2, wcd2);
    float t1 = vec3_dot(wcd2, cross) / denom;
    if (t1 < -tol || t1 > 1.0f + tol) return 0;
    
    vec3_cross(w, d1, wcd1);
    float t2 = vec3_dot(wcd1, cross) / denom;
    if (t2 < -tol || t2 > 1.0f + tol) return 0;
    
    /* Verify intersection points are close */
    float i1[3], i2[3], diff[3];
    vec3_lerp(p1, p2, t1, i1);
    vec3_lerp(p3, p4, t2, i2);
    vec3_sub(i1, i2, diff);
    if (vec3_length_sq(diff) >= tol_sq) return 0;
    
    *t1_out = t1;
    *t2_out = t2;
    pos_out[0] = i1[0];
    pos_out[1] = i1[1];
    pos_out[2] = i1[2];
    return 1;
}

/* ============================================================================
 * Exported Functions
 * ============================================================================ */

/**
 * Find all intersections between panel centerlines.
 * 
 * @param starts  Float array of start points [x0,y0,z0, x1,y1,z1, ...]
 * @param ends    Float array of end points [x0,y0,z0, x1,y1,z1, ...]
 * @param num_panels Number of panels
 * @return Number of intersections found (results in global buffer)
 */
EXPORT int solve_intersections(
    const float* starts,
    const float* ends,
    int num_panels
) {
    g_num_intersections = 0;
    
    for (int i = 0; i < num_panels && g_num_intersections < MAX_INTERSECTIONS; i++) {
        for (int j = i + 1; j < num_panels && g_num_intersections < MAX_INTERSECTIONS; j++) {
            float t_i, t_j, pos[3];
            
            if (segment_intersect(
                &starts[i * 3], &ends[i * 3],
                &starts[j * 3], &ends[j * 3],
                TOLERANCE, &t_i, &t_j, pos
            )) {
                IntersectionResult* r = &g_intersections[g_num_intersections++];
                r->panel_i = i;
                r->panel_j = j;
                r->t_i = t_i;
                r->t_j = t_j;
                r->pos_x = pos[0];
                r->pos_y = pos[1];
                r->pos_z = pos[2];
            }
        }
    }
    
    return g_num_intersections;
}

/**
 * Get pointer to intersection results buffer.
 * Each result is 7 floats: [panel_i, panel_j, t_i, t_j, pos_x, pos_y, pos_z]
 */
EXPORT IntersectionResult* get_intersections_ptr(void) {
    return g_intersections;
}

/**
 * Build simple box geometry for each panel (for testing).
 * Full geometry building would include face segmentation.
 * 
 * @param starts     Float array of start points
 * @param ends       Float array of end points  
 * @param thicknesses Float array of panel thicknesses
 * @param heights    Float array of panel heights
 * @param num_panels Number of panels
 * @return Number of vertices generated
 */
EXPORT int build_geometry(
    const float* starts,
    const float* ends,
    const float* thicknesses,
    const float* heights,
    int num_panels
) {
    g_num_vertices = 0;
    g_num_indices = 0;
    
    for (int p = 0; p < num_panels && g_num_vertices < MAX_PANELS * 24 - 8; p++) {
        float sx = starts[p * 3];
        float sy = starts[p * 3 + 1];
        float sz = starts[p * 3 + 2];
        float ex = ends[p * 3];
        float ey = ends[p * 3 + 1];
        float ez = ends[p * 3 + 2];
        
        float h = heights[p];
        float t = thicknesses[p] * 0.5f;
        
        /* Direction vector */
        float dx = ex - sx;
        float dy = ey - sy;
        float dz = ez - sz;
        float len = sqrtf(dx*dx + dy*dy + dz*dz);
        if (len < TOLERANCE) continue;
        
        /* Normalize */
        dx /= len; dy /= len; dz /= len;
        
        /* Perpendicular (assuming XZ plane for now) */
        float px = -dz;
        float pz = dx;
        
        /* 8 vertices for a box */
        int base = g_num_vertices;
        
        /* Bottom face */
        g_vertices[g_num_vertices * 3 + 0] = sx + px * t;
        g_vertices[g_num_vertices * 3 + 1] = sy;
        g_vertices[g_num_vertices * 3 + 2] = sz + pz * t;
        g_num_vertices++;
        
        g_vertices[g_num_vertices * 3 + 0] = sx - px * t;
        g_vertices[g_num_vertices * 3 + 1] = sy;
        g_vertices[g_num_vertices * 3 + 2] = sz - pz * t;
        g_num_vertices++;
        
        g_vertices[g_num_vertices * 3 + 0] = ex - px * t;
        g_vertices[g_num_vertices * 3 + 1] = ey;
        g_vertices[g_num_vertices * 3 + 2] = ez - pz * t;
        g_num_vertices++;
        
        g_vertices[g_num_vertices * 3 + 0] = ex + px * t;
        g_vertices[g_num_vertices * 3 + 1] = ey;
        g_vertices[g_num_vertices * 3 + 2] = ez + pz * t;
        g_num_vertices++;
        
        /* Top face */
        g_vertices[g_num_vertices * 3 + 0] = sx + px * t;
        g_vertices[g_num_vertices * 3 + 1] = sy + h;
        g_vertices[g_num_vertices * 3 + 2] = sz + pz * t;
        g_num_vertices++;
        
        g_vertices[g_num_vertices * 3 + 0] = sx - px * t;
        g_vertices[g_num_vertices * 3 + 1] = sy + h;
        g_vertices[g_num_vertices * 3 + 2] = sz - pz * t;
        g_num_vertices++;
        
        g_vertices[g_num_vertices * 3 + 0] = ex - px * t;
        g_vertices[g_num_vertices * 3 + 1] = ey + h;
        g_vertices[g_num_vertices * 3 + 2] = ez - pz * t;
        g_num_vertices++;
        
        g_vertices[g_num_vertices * 3 + 0] = ex + px * t;
        g_vertices[g_num_vertices * 3 + 1] = ey + h;
        g_vertices[g_num_vertices * 3 + 2] = ez + pz * t;
        g_num_vertices++;
        
        /* 12 triangles (36 indices) for box faces */
        int faces[36] = {
            /* Bottom */
            0, 1, 2,  0, 2, 3,
            /* Top */
            4, 6, 5,  4, 7, 6,
            /* Front */
            0, 3, 7,  0, 7, 4,
            /* Back */
            1, 5, 6,  1, 6, 2,
            /* Left */
            0, 4, 5,  0, 5, 1,
            /* Right */
            3, 2, 6,  3, 6, 7
        };
        
        for (int i = 0; i < 36; i++) {
            g_indices[g_num_indices++] = base + faces[i];
        }
    }
    
    return g_num_vertices;
}

/**
 * Get pointer to vertices buffer.
 */
EXPORT float* get_vertices_ptr(void) {
    return g_vertices;
}

/**
 * Get pointer to indices buffer.
 */
EXPORT int* get_indices_ptr(void) {
    return g_indices;
}

/**
 * Get number of indices.
 */
EXPORT int get_num_indices(void) {
    return g_num_indices;
}

/**
 * Get number of vertices.
 */
EXPORT int get_num_vertices(void) {
    return g_num_vertices;
}
