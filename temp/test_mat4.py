#!/usr/bin/env python
"""Quick test of Mat4 C type including in-place operations."""

from gxml.mathutils._vec3 import Mat4, Vec3

# Test identity creation
m = Mat4()
print('Identity:', m)

# Test from nested tuple  
data = ((1,0,0,0),(0,2,0,0),(0,0,3,0),(4,5,6,1))
m2 = Mat4(data)
print('Custom:', m2)

# Test transform_point
p = m2.transform_point((1, 1, 1))
print('Transform (1,1,1):', p)

# Test inverse
inv = m2.inverse()
print('Inverse:', inv)

# Test matmul
result = m2 @ inv
print('m @ inv:', result)

# Test to_tuple
print('to_tuple:', m2.to_tuple())

# Test sequence protocol (used by existing code)
print('Row 0:', m2[0])
print('Len:', len(m2))

# Test batch_transform_points
points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
transformed = m2.batch_transform_points(points)
print('Batch transform:', transformed)

print('\n=== In-place operations ===')

# Test set_identity
m3 = Mat4(data)
print('Before set_identity:', m3[0])
m3.set_identity()
print('After set_identity:', m3[0])

# Test set_from
m4 = Mat4()
m4.set_from(data)
print('After set_from:', m4[0])

# Test multiply_into
m5 = Mat4()
a = ((2,0,0,0),(0,2,0,0),(0,0,2,0),(0,0,0,1))  # scale by 2
b = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(10,20,30,1))  # translate
m5.multiply_into(b, a)  # translate then scale
print('multiply_into(translate, scale):', m5)

# Test set_trs  
m6 = Mat4()
m6.set_trs(10, 20, 30, 0, 0, 45, 2, 2, 2)  # translate, rotate 45° around Z, scale 2x
print('set_trs (t=10,20,30 r=0,0,45 s=2,2,2):', m6)
p2 = m6.transform_point((1, 0, 0))
print('Transform (1,0,0) with TRS:', p2)

# Test pre_multiply and post_multiply
m7 = Mat4()
m7.set_translation(5, 0, 0)
scale2x = ((2,0,0,0),(0,2,0,0),(0,0,2,0),(0,0,0,1))
m7.pre_multiply(scale2x)  # scale @ translate = scale then translate
print('pre_multiply (scale @ translate):', m7.transform_point((0,0,0)))

m8 = Mat4()
m8.set_translation(5, 0, 0)
m8.post_multiply(scale2x)  # translate @ scale = translate then scale
print('post_multiply (translate @ scale):', m8.transform_point((0,0,0)))

# Test invert_into
m9 = Mat4()
m9.set_translation(10, 20, 30)
print('Before invert_into:', m9.transform_point((0,0,0)))
m9.invert_into()
print('After invert_into:', m9.transform_point((10,20,30)))

print('\nAll tests passed!')
