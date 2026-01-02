"""
Benchmark GPU vs CPU geometry acceleration.

Tests the Metal GPU accelerator against CPU implementation
for various batch sizes to find the crossover point.
"""

import sys
sys.path.insert(0, '/Users/morgan/Projects/gxml/src')

import numpy as np
import time

from gxml.gpu.metal_geometry import (
    is_metal_available,
    CPUGeometryAccelerator,
    get_accelerator
)

try:
    from gxml.gpu.metal_geometry import MetalGeometryAccelerator
except Exception:
    MetalGeometryAccelerator = None


def generate_test_data(n_matrices: int, n_points_per: int = 4):
    """Generate random test data for benchmarking."""
    # Random rotation matrices (orthogonal, valid transforms)
    matrices = np.zeros((n_matrices, 4, 4), dtype=np.float32)
    for i in range(n_matrices):
        # Random rotation + translation
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        matrices[i] = [
            [c, -s, 0, np.random.uniform(-10, 10)],
            [s, c, 0, np.random.uniform(-10, 10)],
            [0, 0, 1, np.random.uniform(-10, 10)],
            [0, 0, 0, 1]
        ]
    
    # Random local points
    local_points = np.random.uniform(-1, 1, (n_matrices, n_points_per, 4)).astype(np.float32)
    local_points[:, :, 3] = 1.0  # w = 1
    
    return matrices, local_points


def generate_face_point_data(n_points: int):
    """Generate test data for face point computation."""
    matrices = np.zeros((n_points, 4, 4), dtype=np.float32)
    for i in range(n_points):
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        matrices[i] = [
            [c, -s, 0, np.random.uniform(-10, 10)],
            [s, c, 0, np.random.uniform(-10, 10)],
            [0, 0, 1, np.random.uniform(-10, 10)],
            [0, 0, 0, 1]
        ]
    
    half_thicknesses = np.random.uniform(0.01, 0.1, n_points).astype(np.float32)
    ts_coords = np.random.uniform(0, 1, (n_points, 2)).astype(np.float32)
    face_sides = np.random.randint(0, 6, n_points).astype(np.uint32)
    
    return matrices, half_thicknesses, ts_coords, face_sides


def benchmark_transform_points(accel, matrices, local_points, n_iters: int = 100):
    """Benchmark point transformation."""
    # Warmup
    for _ in range(5):
        accel.transform_points_batch(matrices, local_points)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(n_iters):
        result = accel.transform_points_batch(matrices, local_points)
    elapsed = time.perf_counter() - start
    
    return elapsed / n_iters, result


def benchmark_face_points(accel, matrices, half_t, ts, sides, n_iters: int = 100):
    """Benchmark face point computation."""
    # Warmup
    for _ in range(5):
        accel.compute_face_points_batch(matrices, half_t, ts, sides)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(n_iters):
        result = accel.compute_face_points_batch(matrices, half_t, ts, sides)
    elapsed = time.perf_counter() - start
    
    return elapsed / n_iters, result


def verify_correctness():
    """Verify GPU and CPU produce same results."""
    print("Verifying correctness...")
    
    cpu = CPUGeometryAccelerator()
    
    if is_metal_available():
        gpu = MetalGeometryAccelerator()
        
        # Test transform_points_batch
        matrices, local_points = generate_test_data(100, 4)
        cpu_result = cpu.transform_points_batch(matrices, local_points)
        gpu_result = gpu.transform_points_batch(matrices, local_points)
        
        if np.allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-4):
            print("  ✓ transform_points_batch: PASS")
        else:
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            print(f"  ✗ transform_points_batch: FAIL (max diff: {max_diff:.6e})")
        
        # Test compute_face_points_batch
        matrices, half_t, ts, sides = generate_face_point_data(1000)
        cpu_result = cpu.compute_face_points_batch(matrices, half_t, ts, sides)
        gpu_result = gpu.compute_face_points_batch(matrices, half_t, ts, sides)
        
        if np.allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-4):
            print("  ✓ compute_face_points_batch: PASS")
        else:
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            print(f"  ✗ compute_face_points_batch: FAIL (max diff: {max_diff:.6e})")
        
        # Test intersection
        n_tests = 1000
        lines_a = np.random.uniform(-10, 10, (n_tests, 4)).astype(np.float32)
        lines_b = np.random.uniform(-10, 10, (n_tests, 4)).astype(np.float32)
        
        cpu_result = cpu.test_intersections_2d(lines_a, lines_b)
        gpu_result = gpu.test_intersections_2d(lines_a, lines_b)
        
        if np.allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-4):
            print("  ✓ test_intersections_2d: PASS")
        else:
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            print(f"  ✗ test_intersections_2d: FAIL (max diff: {max_diff:.6e})")
    else:
        print("  Metal not available, skipping GPU verification")
    
    print()


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 70)
    print("GXML GPU Acceleration Benchmark")
    print("=" * 70)
    print()
    
    print(f"Metal GPU available: {is_metal_available()}")
    print()
    
    verify_correctness()
    
    cpu = CPUGeometryAccelerator()
    gpu = MetalGeometryAccelerator() if is_metal_available() else None
    
    # Benchmark transform_points_batch for various sizes
    print("=" * 70)
    print("Benchmark: transform_points_batch")
    print("(N matrices × 4 points each → N*4 transformed points)")
    print("=" * 70)
    print()
    print(f"{'N Matrices':<12} {'N Points':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<12}")
    print("-" * 60)
    
    for n_matrices in [100, 500, 1000, 2000, 5000, 10000]:
        matrices, local_points = generate_test_data(n_matrices, 4)
        n_points = n_matrices * 4
        
        cpu_time, _ = benchmark_transform_points(cpu, matrices, local_points, n_iters=50)
        
        if gpu:
            gpu_time, _ = benchmark_transform_points(gpu, matrices, local_points, n_iters=50)
            speedup = cpu_time / gpu_time
            print(f"{n_matrices:<12} {n_points:<12} {cpu_time*1000:>10.3f}  {gpu_time*1000:>10.3f}  {speedup:>10.1f}x")
        else:
            print(f"{n_matrices:<12} {n_points:<12} {cpu_time*1000:>10.3f}  {'N/A':>10}  {'N/A':>10}")
    
    print()
    
    # Benchmark compute_face_points_batch
    print("=" * 70)
    print("Benchmark: compute_face_points_batch")
    print("(Combined local point computation + transform)")
    print("=" * 70)
    print()
    print(f"{'N Points':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<12}")
    print("-" * 48)
    
    for n_points in [1000, 5000, 10000, 20000, 50000, 100000]:
        matrices, half_t, ts, sides = generate_face_point_data(n_points)
        
        cpu_time, _ = benchmark_face_points(cpu, matrices, half_t, ts, sides, n_iters=50)
        
        if gpu:
            gpu_time, _ = benchmark_face_points(gpu, matrices, half_t, ts, sides, n_iters=50)
            speedup = cpu_time / gpu_time
            print(f"{n_points:<12} {cpu_time*1000:>10.3f}  {gpu_time*1000:>10.3f}  {speedup:>10.1f}x")
        else:
            print(f"{n_points:<12} {cpu_time*1000:>10.3f}  {'N/A':>10}  {'N/A':>10}")
    
    print()
    
    # GXML-scale estimate
    print("=" * 70)
    print("GXML 75-Panel Layout Estimate")
    print("=" * 70)
    print()
    print("75 panels × ~15 segments/panel × 4 corners = ~4500 transforms")
    print()
    
    n_transforms = 4500
    matrices, local_points = generate_test_data(n_transforms // 4, 4)
    
    cpu_time, _ = benchmark_transform_points(cpu, matrices, local_points, n_iters=100)
    print(f"CPU time for {n_transforms} transforms: {cpu_time*1000:.3f} ms")
    
    if gpu:
        gpu_time, _ = benchmark_transform_points(gpu, matrices, local_points, n_iters=100)
        print(f"GPU time for {n_transforms} transforms: {gpu_time*1000:.3f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
    print()


if __name__ == "__main__":
    run_benchmarks()
