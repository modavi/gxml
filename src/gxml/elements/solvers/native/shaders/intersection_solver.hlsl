// GXML Intersection Solver - DirectX 12 HLSL Compute Shader
// Finds intersections between panel centerlines in parallel
//
// Compile with: dxc -T cs_6_0 -E find_intersections intersection_solver.hlsl -Fo intersection_solver.cso

// ============================================================================
// Structures
// ============================================================================

struct Panel
{
    float3 start;
    float _pad0;
    float3 end;
    float _pad1;
};

struct IntersectionResult
{
    uint panel_i;
    uint panel_j;
    float t_i;
    float t_j;
    float3 position;
    uint valid;
};

struct Uniforms
{
    uint num_panels;
    float tolerance;
    float2 _pad;
};

// ============================================================================
// Resources
// ============================================================================

StructuredBuffer<Panel> panels : register(t0);
RWStructuredBuffer<IntersectionResult> results : register(u0);
RWByteAddressBuffer result_count : register(u1);  // Atomic counter
ConstantBuffer<Uniforms> uniforms : register(b0);

// ============================================================================
// Helper Functions
// ============================================================================

float3 vec3_cross(float3 a, float3 b)
{
    return float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

bool segment_intersect(
    float3 p1, float3 p2,
    float3 p3, float3 p4,
    float tol,
    out float t1_out,
    out float t2_out,
    out float3 pos_out
)
{
    float3 d1 = p2 - p1;
    float3 d2 = p4 - p3;
    float3 w = p3 - p1;
    
    float3 cross_d = vec3_cross(d1, d2);
    float denom = dot(cross_d, cross_d);
    
    float tol_sq = tol * tol;
    if (denom < tol_sq)
    {
        return false;  // Parallel lines
    }
    
    float3 wcd2 = vec3_cross(w, d2);
    float t1 = dot(wcd2, cross_d) / denom;
    if (t1 < -tol || t1 > 1.0 + tol)
    {
        return false;
    }
    
    float3 wcd1 = vec3_cross(w, d1);
    float t2 = dot(wcd1, cross_d) / denom;
    if (t2 < -tol || t2 > 1.0 + tol)
    {
        return false;
    }
    
    // Verify intersection points are close
    float3 i1 = lerp(p1, p2, t1);
    float3 i2 = lerp(p3, p4, t2);
    float3 diff = i1 - i2;
    if (dot(diff, diff) >= tol_sq)
    {
        return false;
    }
    
    t1_out = t1;
    t2_out = t2;
    pos_out = i1;
    return true;
}

// ============================================================================
// Intersection Solver Kernel
// ============================================================================

[numthreads(8, 8, 1)]
void find_intersections(uint3 gid : SV_DispatchThreadID)
{
    uint i = gid.x;
    uint j = gid.y;
    
    // Only process upper triangle (i < j)
    if (i >= j || j >= uniforms.num_panels)
    {
        return;
    }
    
    Panel panel_i = panels[i];
    Panel panel_j = panels[j];
    
    float t_i, t_j;
    float3 pos;
    
    if (segment_intersect(
        panel_i.start, panel_i.end,
        panel_j.start, panel_j.end,
        uniforms.tolerance,
        t_i, t_j, pos
    ))
    {
        // Atomically increment counter and get index
        uint idx;
        result_count.InterlockedAdd(0, 1, idx);
        
        // Store result
        IntersectionResult result;
        result.panel_i = i;
        result.panel_j = j;
        result.t_i = t_i;
        result.t_j = t_j;
        result.position = pos;
        result.valid = 1;
        
        results[idx] = result;
    }
}

// ============================================================================
// Batch Transform Kernel
// ============================================================================

struct TransformInput
{
    float4x4 matrix;
    float4 point;
};

StructuredBuffer<TransformInput> transforms : register(t1);
RWStructuredBuffer<float4> output_points : register(u2);

[numthreads(64, 1, 1)]
void transform_points(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    TransformInput input = transforms[idx];
    output_points[idx] = mul(input.matrix, input.point);
}

// ============================================================================
// Face Point Computation Kernel
// ============================================================================

struct FacePointInput
{
    float4x4 matrix;
    float half_thickness;
    float t;
    float s;
    uint face_side;  // 0=FRONT, 1=BACK, 2=TOP, 3=BOTTOM, 4=START, 5=END
};

StructuredBuffer<FacePointInput> face_inputs : register(t2);
RWStructuredBuffer<float4> world_points : register(u3);

[numthreads(64, 1, 1)]
void compute_face_points(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    FacePointInput input = face_inputs[idx];
    
    float half_t = input.half_thickness;
    float t = input.t;
    float s = input.s;
    
    float4 local;
    
    switch (input.face_side)
    {
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
    
    world_points[idx] = mul(input.matrix, local);
}

// ============================================================================
// 2D Line Intersection Test (for UI/debugging)
// ============================================================================

struct Line2D
{
    float4 coords;  // (x1, y1, x2, y2)
};

StructuredBuffer<Line2D> lines_a : register(t3);
StructuredBuffer<Line2D> lines_b : register(t4);
RWStructuredBuffer<float4> intersection_results : register(u4);  // (t_a, t_b, x, y)

[numthreads(64, 1, 1)]
void test_intersections_2d(uint3 gid : SV_DispatchThreadID)
{
    uint idx = gid.x;
    
    float4 a = lines_a[idx].coords;
    float4 b = lines_b[idx].coords;
    
    // Line A: P = A1 + t*(A2-A1)
    float ax = a.z - a.x;
    float ay = a.w - a.y;
    float bx = b.z - b.x;
    float by = b.w - b.y;
    
    float denom = ax * by - ay * bx;
    
    if (abs(denom) < 1e-10)
    {
        // Parallel lines
        intersection_results[idx] = float4(-1.0, -1.0, 0.0, 0.0);
        return;
    }
    
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    
    float t = (dx * by - dy * bx) / denom;
    float s = (dx * ay - dy * ax) / denom;
    
    // Intersection point
    float ix = a.x + t * ax;
    float iy = a.y + t * ay;
    
    intersection_results[idx] = float4(t, s, ix, iy);
}
