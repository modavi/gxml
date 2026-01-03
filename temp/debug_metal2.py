"""Try different Taichi Metal configurations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gxml'))

import taichi as ti

print("Taichi version:", ti.__version__)

# Try with offline_cache disabled (sometimes helps with shader compilation)
print("\n--- Attempt 1: offline_cache=False ---")
try:
    ti.reset()
    ti.init(arch=ti.metal, offline_cache=False)
    
    x = ti.field(dtype=ti.f32, shape=10)
    
    @ti.kernel
    def test1():
        for i in range(10):
            x[i] = float(i)
    
    test1()
    ti.sync()
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")

# Try with default_fp=ti.f32 explicitly
print("\n--- Attempt 2: default_fp=ti.f32 ---")
try:
    ti.reset()
    ti.init(arch=ti.metal, default_fp=ti.f32, offline_cache=False)
    
    y = ti.field(dtype=ti.f32, shape=10)
    
    @ti.kernel
    def test2():
        for i in range(10):
            y[i] = float(i)
    
    test2()
    ti.sync()
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")

# Try with device_memory_fraction
print("\n--- Attempt 3: smaller device memory ---")
try:
    ti.reset()
    ti.init(arch=ti.metal, device_memory_fraction=0.5, offline_cache=False)
    
    z = ti.field(dtype=ti.f32, shape=10)
    
    @ti.kernel
    def test3():
        for i in range(10):
            z[i] = float(i)
    
    test3()
    ti.sync()
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")

# Try vulkan instead (might work on macOS via MoltenVK)
print("\n--- Attempt 4: Vulkan backend ---")
try:
    ti.reset()
    ti.init(arch=ti.vulkan)
    
    w = ti.field(dtype=ti.f32, shape=10)
    
    @ti.kernel
    def test4():
        for i in range(10):
            w[i] = float(i)
    
    test4()
    ti.sync()
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")

# Check available backends
print("\n--- Available backends ---")
print(f"Metal supported: {ti.lang.impl.is_extension_supported(ti.extension.metal, ti.metal)}")
try:
    print(f"Vulkan supported: {ti.lang.impl.is_extension_supported(ti.extension.spirv, ti.vulkan)}")
except:
    print("Vulkan: not available")
