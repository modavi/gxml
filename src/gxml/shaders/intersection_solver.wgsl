// GXML Intersection Solver - WebGPU Compute Shader
// Finds intersections between panel centerlines in parallel
//
// This shader tests all panel pairs for intersection in O(n²) work
// distributed across GPU threads. Each thread handles one pair.

// ============================================================================
// Structures
// ============================================================================

struct Panel {
    start: vec3f,
    _pad0: f32,  // Padding for 16-byte alignment
    end: vec3f,
    _pad1: f32,
}

struct IntersectionResult {
    panel_i: u32,
    panel_j: u32,
    t_i: f32,
    t_j: f32,
    position: vec3f,
    valid: u32,  // 1 if intersection found, 0 otherwise
}

struct Uniforms {
    num_panels: u32,
    tolerance: f32,
    _pad: vec2f,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> panels: array<Panel>;
@group(0) @binding(1) var<storage, read_write> results: array<IntersectionResult>;
@group(0) @binding(2) var<storage, read_write> result_count: atomic<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

// ============================================================================
// Helper Functions
// ============================================================================

fn vec3_cross(a: vec3f, b: vec3f) -> vec3f {
    return vec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

fn segment_intersect(
    p1: vec3f, p2: vec3f,
    p3: vec3f, p4: vec3f,
    tol: f32,
    t1_out: ptr<function, f32>,
    t2_out: ptr<function, f32>,
    pos_out: ptr<function, vec3f>
) -> bool {
    let d1 = p2 - p1;
    let d2 = p4 - p3;
    let w = p3 - p1;
    
    let cross_d = vec3_cross(d1, d2);
    let denom = dot(cross_d, cross_d);
    
    let tol_sq = tol * tol;
    if (denom < tol_sq) {
        return false;  // Parallel lines
    }
    
    let wcd2 = vec3_cross(w, d2);
    let t1 = dot(wcd2, cross_d) / denom;
    if (t1 < -tol || t1 > 1.0 + tol) {
        return false;
    }
    
    let wcd1 = vec3_cross(w, d1);
    let t2 = dot(wcd1, cross_d) / denom;
    if (t2 < -tol || t2 > 1.0 + tol) {
        return false;
    }
    
    // Verify intersection points are close
    let i1 = mix(p1, p2, t1);
    let i2 = mix(p3, p4, t2);
    let diff = i1 - i2;
    if (dot(diff, diff) >= tol_sq) {
        return false;
    }
    
    *t1_out = t1;
    *t2_out = t2;
    *pos_out = i1;
    return true;
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn find_intersections(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let j = gid.y;
    
    // Only process upper triangle (i < j)
    if (i >= j || j >= uniforms.num_panels) {
        return;
    }
    
    let panel_i = panels[i];
    let panel_j = panels[j];
    
    var t_i: f32;
    var t_j: f32;
    var pos: vec3f;
    
    if (segment_intersect(
        panel_i.start, panel_i.end,
        panel_j.start, panel_j.end,
        uniforms.tolerance,
        &t_i, &t_j, &pos
    )) {
        // Atomically get next result slot
        let idx = atomicAdd(&result_count, 1u);
        
        // Store result
        results[idx].panel_i = i;
        results[idx].panel_j = j;
        results[idx].t_i = t_i;
        results[idx].t_j = t_j;
        results[idx].position = pos;
        results[idx].valid = 1u;
    }
}

// ============================================================================
// Batch Transform Kernel
// ============================================================================

struct TransformInput {
    matrix: mat4x4f,
    point: vec4f,
}

@group(0) @binding(0) var<storage, read> transforms: array<TransformInput>;
@group(0) @binding(1) var<storage, read_write> output_points: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn transform_points(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let input = transforms[idx];
    output_points[idx] = input.matrix * input.point;
}

// ============================================================================
// Face Point Computation Kernel
// ============================================================================

struct FacePointInput {
    matrix: mat4x4f,
    half_thickness: f32,
    t: f32,
    s: f32,
    face_side: u32,  // 0=FRONT, 1=BACK, 2=TOP, 3=BOTTOM, 4=START, 5=END
}

@group(0) @binding(0) var<storage, read> face_inputs: array<FacePointInput>;
@group(0) @binding(1) var<storage, read_write> world_points: array<vec4f>;

@compute @workgroup_size(64, 1, 1)
fn compute_face_points(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let input = face_inputs[idx];
    
    let half_t = input.half_thickness;
    let t = input.t;
    let s = input.s;
    
    var local: vec4f;
    
    switch (input.face_side) {
        case 0u: {  // FRONT
            local = vec4f(t, s, half_t, 1.0);
        }
        case 1u: {  // BACK
            local = vec4f(t, s, -half_t, 1.0);
        }
        case 2u: {  // TOP
            local = vec4f(t, 1.0, -half_t + s * half_t * 2.0, 1.0);
        }
        case 3u: {  // BOTTOM
            local = vec4f(t, 0.0, -half_t + s * half_t * 2.0, 1.0);
        }
        case 4u: {  // START
            local = vec4f(0.0, s, -half_t + t * half_t * 2.0, 1.0);
        }
        case 5u: {  // END
            local = vec4f(1.0, s, -half_t + t * half_t * 2.0, 1.0);
        }
        default: {
            local = vec4f(t, s, 0.0, 1.0);
        }
    }
    
    world_points[idx] = input.matrix * local;
}
