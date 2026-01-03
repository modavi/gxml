// GXML Intersection Solver - Metal Compute Shader
// Finds intersections between panel centerlines in parallel
//
// This is a standalone version of the shaders embedded in metal_geometry.py
// for use with the cross-platform shader backend.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Transform Kernels
// ============================================================================

// Transform a batch of points by a batch of matrices
kernel void transform_points_batch(
    device const float4x4* matrices [[buffer(0)]],
    device const float4* local_points [[buffer(1)]],
    device float4* world_points [[buffer(2)]],
    device const uint* points_per_matrix [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint num_points = *points_per_matrix;
    uint matrix_idx = gid / num_points;
    uint point_idx = gid % num_points;
    
    float4x4 mat = matrices[matrix_idx];
    float4 local = local_points[matrix_idx * num_points + point_idx];
    
    float4 world;
    world.x = dot(mat[0], local);
    world.y = dot(mat[1], local);
    world.z = dot(mat[2], local);
    world.w = dot(mat[3], local);
    
    world_points[gid] = world;
}

// ============================================================================
// Face Point Computation
// ============================================================================

kernel void compute_face_points(
    device const float4x4* matrices [[buffer(0)]],
    device const float* half_thicknesses [[buffer(1)]],
    device const float2* ts_coords [[buffer(2)]],
    device const uint* face_sides [[buffer(3)]],
    device float4* world_points [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float4x4 mat = matrices[gid];
    float half_t = half_thicknesses[gid];
    float2 ts = ts_coords[gid];
    uint side = face_sides[gid];
    
    float t = ts.x;
    float s = ts.y;
    float4 local;
    
    switch (side) {
        case 0:  // FRONT
            local = float4(t, s, half_t, 1.0);
            break;
        case 1:  // BACK
            local = float4(t, s, -half_t, 1.0);
            break;
        case 2:  // TOP
            local = float4(t, 1.0, -half_t + s * half_t * 2.0, 1.0);
            break;
        case 3:  // BOTTOM
            local = float4(t, 0.0, -half_t + s * half_t * 2.0, 1.0);
            break;
        case 4:  // START
            local = float4(0.0, s, -half_t + t * half_t * 2.0, 1.0);
            break;
        case 5:  // END
            local = float4(1.0, s, -half_t + t * half_t * 2.0, 1.0);
            break;
        default:
            local = float4(t, s, 0.0, 1.0);
            break;
    }
    
    float4 world;
    world.x = dot(mat[0], local);
    world.y = dot(mat[1], local);
    world.z = dot(mat[2], local);
    world.w = dot(mat[3], local);
    
    world_points[gid] = world;
}

// ============================================================================
// Intersection Solver
// ============================================================================

struct Panel {
    float3 start;
    float _pad0;
    float3 end;
    float _pad1;
};

struct IntersectionResult {
    uint panel_i;
    uint panel_j;
    float t_i;
    float t_j;
    float3 position;
    uint valid;
};

struct Uniforms {
    uint num_panels;
    float tolerance;
    float2 _pad;
};

kernel void find_intersections(
    device const Panel* panels [[buffer(0)]],
    device IntersectionResult* results [[buffer(1)]],
    device atomic_uint* result_count [[buffer(2)]],
    constant Uniforms& uniforms [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    
    if (i >= j || j >= uniforms.num_panels) {
        return;
    }
    
    float3 p1 = panels[i].start;
    float3 p2 = panels[i].end;
    float3 p3 = panels[j].start;
    float3 p4 = panels[j].end;
    
    float3 d1 = p2 - p1;
    float3 d2 = p4 - p3;
    float3 w = p3 - p1;
    
    float3 cross_d = cross(d1, d2);
    float denom = dot(cross_d, cross_d);
    
    float tol = uniforms.tolerance;
    float tol_sq = tol * tol;
    
    if (denom < tol_sq) {
        return;
    }
    
    float3 wcd2 = cross(w, d2);
    float t1 = dot(wcd2, cross_d) / denom;
    if (t1 < -tol || t1 > 1.0 + tol) {
        return;
    }
    
    float3 wcd1 = cross(w, d1);
    float t2 = dot(wcd1, cross_d) / denom;
    if (t2 < -tol || t2 > 1.0 + tol) {
        return;
    }
    
    float3 i1 = mix(p1, p2, t1);
    float3 i2 = mix(p3, p4, t2);
    float3 diff = i1 - i2;
    if (dot(diff, diff) >= tol_sq) {
        return;
    }
    
    uint idx = atomic_fetch_add_explicit(result_count, 1, memory_order_relaxed);
    
    results[idx].panel_i = i;
    results[idx].panel_j = j;
    results[idx].t_i = t1;
    results[idx].t_j = t2;
    results[idx].position = i1;
    results[idx].valid = 1;
}

// ============================================================================
// 2D Line Intersection (for debugging/UI)
// ============================================================================

kernel void test_intersections_2d(
    device const float4* lines_a [[buffer(0)]],
    device const float4* lines_b [[buffer(1)]],
    device float4* results [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float4 a = lines_a[gid];
    float4 b = lines_b[gid];
    
    float ax = a.z - a.x;
    float ay = a.w - a.y;
    float bx = b.z - b.x;
    float by = b.w - b.y;
    
    float denom = ax * by - ay * bx;
    
    if (abs(denom) < 1e-10) {
        results[gid] = float4(-1.0, -1.0, 0.0, 0.0);
        return;
    }
    
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    
    float t = (dx * by - dy * bx) / denom;
    float s = (dx * ay - dy * ax) / denom;
    
    float ix = a.x + t * ax;
    float iy = a.y + t * ay;
    
    results[gid] = float4(t, s, ix, iy);
}
