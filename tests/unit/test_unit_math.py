"""
Unit tests for GXMLMath utility functions.

Tests cover matrix operations (creation, multiplication, inversion), 
vector operations (normalization, cross product, dot product), 
geometric intersections (ray-segment, ray-ray), and transform utilities.
"""

import unittest
import numpy as np
import mathutils.gxml_math as GXMLMath
from mathutils.gxml_ray import GXMLRay


class GXMLMathMatrixTests(unittest.TestCase):
    """Tests for matrix creation and manipulation operations"""
    
    def testIdentityMatrix(self):
        """Test identity matrix creation"""
        identity_matrix = GXMLMath.identity()
        expected = np.array([[1., 0., 0., 0.], 
                           [0., 1., 0., 0.], 
                           [0., 0., 1., 0.], 
                           [0., 0., 0., 1.]])
        self.assertTrue(np.array_equal(identity_matrix, expected), 
                       "Identity matrix should match expected values")
        
    def testCreatetranslate_matrix(self):
        """Test translation matrix creation"""
        translate_matrix = GXMLMath.translate_matrix((5, 9, 7))
        expected = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [5, 9, 7, 1]])
        self.assertTrue(np.allclose(translate_matrix, expected),
                       "Translation matrix should have translation in bottom row")
        
    def testCreateRotationMatrix(self):
        """Test rotation matrix creation with different rotation orders"""
        # Default rotation order
        rotation_matrix = GXMLMath.rot_matrix((55, 66, 77))
        expected = np.array([[0.0914958, 0.396312, -0.913545, 0],
                           [-0.390537, 0.85818, 0.333179, 0],
                           [0.916029, 0.326289, 0.233295, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "Rotation matrix with default order should match expected")
        
        # XYZ rotation order
        rotation_matrix = GXMLMath.rot_matrix((23, 36, 92), rotate_order="xyz")
        expected = np.array([[-0.0282343, 0.808524, -0.587785, 0],
                           [-0.927959, 0.197401, 0.316108, 0],
                           [0.37161, 0.554366, 0.744704, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "Rotation matrix with xyz order should match expected")
        
        # ZYX rotation order
        rotation_matrix = GXMLMath.rot_matrix((23, 36, 92), rotate_order="zyx")
        expected = np.array([[-0.0282343, 0.911929, 0.409376, 0],
                           [-0.808524, -0.261651, 0.527093, 0],
                           [0.587785, -0.316108, 0.744704, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "Rotation matrix with zyx order should match expected")
        
        # YZX rotation order
        rotation_matrix = GXMLMath.rot_matrix((23, 36, 92), rotate_order="yzx")
        expected = np.array([[-0.0282343, 0.973916, -0.225144, 0],
                           [-0.999391, -0.0321252, -0.0136363, 0],
                           [-0.0205134, 0.224621, 0.97423, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "Rotation matrix with yzx order should match expected")
        
    def testCreatescale_matrix(self):
        """Test scale matrix creation"""
        scale_matrix = GXMLMath.scale_matrix((4.2, 3.7, 9.5))
        expected = np.array([[4.2, 0, 0, 0],
                           [0, 3.7, 0, 0],
                           [0, 0, 9.5, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(scale_matrix, expected),
                       "Scale matrix should have scale values on diagonal")
        
    def testCreateTranslateTransformMatrix(self):
        """Test combined transform matrix with only translation"""
        translate_matrix = GXMLMath.build_transform_matrix((5, 9, 7), (0, 0, 0), (1, 1, 1))
        expected = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [5, 9, 7, 1]])
        self.assertTrue(np.allclose(translate_matrix, expected),
                       "Transform with only translation should match translate matrix")
        
    def testCreateRotationTransformMatrix(self):
        """Test combined transform matrix with only rotation"""
        rotation_matrix = GXMLMath.build_transform_matrix((0, 0, 0), (23, 36, 92), (1, 1, 1))
        expected = np.array([[-0.0282343, 0.808524, -0.587785, 0],
                           [-0.927959, 0.197401, 0.316108, 0],
                           [0.37161, 0.554366, 0.744704, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "Transform with only rotation should match rotation matrix")
        
    def testCreateScaleTransformMatrix(self):
        """Test combined transform matrix with only scale"""
        scale_matrix = GXMLMath.build_transform_matrix((0, 0, 0), (0, 0, 0), (4.2, 3.7, 9.5))
        expected = np.array([[4.2, 0, 0, 0],
                           [0, 3.7, 0, 0],
                           [0, 0, 9.5, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(scale_matrix, expected),
                       "Transform with only scale should match scale matrix")
        
    def testCreateTransformMatrix(self):
        """Test combined transform matrix with translation, rotation, and scale"""
        transform_matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        expected = np.array([[-0.118584, 3.3958, -2.4687, 0],
                           [-3.43345, 0.730383, 1.1696, 0],
                           [3.5303, 5.26648, 7.07469, 0],
                           [5, 9, 7, 1]])
        self.assertTrue(np.allclose(transform_matrix, expected),
                       "Combined transform matrix should match expected values")
        
    def testcombine_transform_matrix(self):
        """Test combining pre-computed transform matrices with different orders"""
        t = GXMLMath.translate_matrix((5, 9, 7))
        r = GXMLMath.rot_matrix((23, 36, 92))
        s = GXMLMath.scale_matrix((2, 3, 4))
        
        # Test SRT order (default)
        actual = GXMLMath.combine_transform_matrix(t, r, s, transform_order="srt")
        expected = GXMLMath.mat_mul(s, GXMLMath.mat_mul(r, t))
        self.assertTrue(np.allclose(actual, expected),
                       "SRT order should multiply Scale * Rotation * Translation")
        
        # Test TRS order
        actual = GXMLMath.combine_transform_matrix(t, r, s, transform_order="trs")
        expected = GXMLMath.mat_mul(t, GXMLMath.mat_mul(r, s))
        self.assertTrue(np.allclose(actual, expected),
                       "TRS order should multiply Translation * Rotation * Scale")
        
        # Test RST order
        actual = GXMLMath.combine_transform_matrix(t, r, s, transform_order="rst")
        expected = GXMLMath.mat_mul(r, GXMLMath.mat_mul(s, t))
        self.assertTrue(np.allclose(actual, expected),
                       "RST order should multiply Rotation * Scale * Translation")
    
    def testMatrixMultiply(self):
        """Test matrix multiplication in different orders"""
        translate_matrix = GXMLMath.translate_matrix((5, 9, 7))
        rotation_matrix = GXMLMath.rot_matrix((23, 36, 92))
        
        # Translate * Rotate
        actual = GXMLMath.mat_mul(translate_matrix, rotation_matrix)
        expected = np.array([[-0.0282343, 0.808524, -0.587785, 0],
                           [-0.927959, 0.197401, 0.316108, 0],
                           [0.37161, 0.554366, 0.744704, 0],
                           [-5.89153, 9.69979, 5.11898, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "Translate * Rotate should apply rotation then move origin")
        
        # Rotate * Translate
        actual = GXMLMath.mat_mul(rotation_matrix, translate_matrix)
        expected = np.array([[-0.0282343, 0.808524, -0.587785, 0],
                           [-0.927959, 0.197401, 0.316108, 0],
                           [0.37161, 0.554366, 0.744704, 0],
                           [5, 9, 7, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "Rotate * Translate should preserve translation")
        
    def testInvert(self):
        """Test matrix inversion"""
        matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        
        actual = GXMLMath.invert(matrix)
        expected = np.array([[-0.00672245, -0.2508, 0.0391169, 0],
                           [0.192506, 0.0533516, 0.0583543, 0],
                           [-0.139949, 0.0854346, 0.0783899, 0],
                           [-0.719297, 0.175792, -1.2695, 1]])
        
        self.assertTrue(np.allclose(actual, expected),
                       "Inverted matrix should match expected values")
    
    def testTranspose(self):
        """Test matrix transpose"""
        matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        
        actual = GXMLMath.transpose(matrix)
        expected = np.array([[-0.118584, -3.43345, 3.5303, 5],
                           [3.3958, 0.730383, 5.26648, 9],
                           [-2.4687, 1.1696, 7.07469, 7],
                           [0, 0, 0, 1]])
        
        self.assertTrue(np.allclose(actual, expected),
                       "Transposed matrix should swap rows and columns")
    
    def testexplode_matrix(self):
        """Test decomposing matrix into translation, rotation, and scale"""
        matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        t, r, s = GXMLMath.explode_matrix(matrix)
        
        self.assertTrue(np.allclose(t, (5, 9, 7)), "Translation should match input")
        self.assertTrue(np.allclose(s, (4.2, 3.7, 9.5)), "Scale should match input")
        self.assertTrue(np.allclose(r, np.array([[-0.0282343, 0.808524, -0.587785],
                                                [-0.927959, 0.197401, 0.316108],
                                                [0.37161, 0.554366, 0.744704]])),
                       "Rotation matrix should match expected 3x3 matrix")
        
    def testExplodeInvalidMatrix(self):
        """Test that exploding invalid matrix raises ValueError"""
        matrix = np.array([[1, 2, 3, 4, 5],
                         [5, 6, 7, 8, 3],
                         [9, 10, 11, 12, 12],
                         [13, 14, 15, 16, 4],
                         [9, 14, 15, 16, 4]])
        with self.assertRaises(ValueError):
            GXMLMath.explode_matrix(matrix)
    
    def testextract_euler_rotation(self):
        """Test extracting Euler angles from transform matrix"""
        matrix = GXMLMath.build_transform_matrix((1, 1, 1), (22, 36, 49), (1, 1, 1))
        actual = GXMLMath.extract_euler_rotation(matrix)
        expected = (22, 36, 49)
        self.assertTrue(np.allclose(actual, expected),
                       "Extracted Euler angles should match input angles")
    
    def testextract_euler_rotationInvalid(self):
        """Test that extracting Euler angles from invalid matrix raises ValueError"""
        matrix = np.array([[1, 2, 3, 4, 5],
                         [5, 6, 7, 8, 3],
                         [9, 10, 11, 12, 12],
                         [13, 14, 15, 16, 4],
                         [9, 14, 15, 16, 4]])
        with self.assertRaises(ValueError):
            GXMLMath.extract_euler_rotation(matrix)
    
    def testMat3ToMat4(self):
        """Test converting 3x3 matrix to 4x4 matrix"""
        mat3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mat4 = GXMLMath.mat3_to_4(mat3)
        expected = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(mat4, expected),
                       "4x4 matrix should have 3x3 in top-left, identity in bottom-right")
    
    def testMatrixDeterminant(self):
        """Test matrix determinant calculation"""
        # Generic matrix
        matrix = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]])
        
        actual = GXMLMath.determinant(matrix)
        expected = np.linalg.det(matrix)
        self.assertTrue(np.allclose(actual, expected),
                       "Determinant should match numpy's calculation")
        
        # Transform matrix
        matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        actual = GXMLMath.determinant(matrix)
        expected = 147.63000000000005
        self.assertTrue(np.allclose(actual, expected),
                       "Transform matrix determinant should match expected")
        
        # Zero scale (singular matrix)
        matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 0, 0))
        actual = GXMLMath.determinant(matrix)
        expected = 0
        self.assertTrue(np.allclose(actual, expected),
                       "Matrix with zero scale should have zero determinant")
    
    def testCreateRotationMatrixFromTriangle(self):
        """Test creating rotation matrix from three points forming a triangle"""
        # Test with simple right triangle in XY plane
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        p2 = np.array([1, 1, 0])
        
        rotation_matrix = np.array(GXMLMath.get_plane_rotation_from_triangle(p0, p1, p2))
        
        # The X axis should point from p1 to p2 (0, 1, 0)
        x_axis = rotation_matrix[:3, 0]
        expected_x = np.array([0, 1, 0])
        self.assertTrue(np.allclose(x_axis, expected_x),
                       "X axis should point from p1 to p2")
        
        # The Y axis should point from p0 to p1 (1, 0, 0)
        y_axis = rotation_matrix[:3, 1]
        expected_y = np.array([1, 0, 0])
        self.assertTrue(np.allclose(y_axis, expected_y),
                       "Y axis should point from p0 to p1")
        
        # The Z axis should be perpendicular (cross product)
        z_axis = rotation_matrix[:3, 2]
        expected_z = np.array([0, 0, -1])
        self.assertTrue(np.allclose(z_axis, expected_z),
                       "Z axis should be cross product of X and Y")
    
    def testCreateRotationMatrixFromAxis(self):
        """Test creating rotation matrix from basis vectors"""
        # Test with standard basis vectors (identity case)
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        rotation_matrix = GXMLMath.get_plane_rotation_from_axis(x_axis, y_axis, z_axis)
        expected = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "Standard basis vectors should produce identity matrix")
        
        # Test with 90-degree rotation around Y axis (X->Z, Y->Y, Z->-X)
        x_axis = np.array([0, 0, 1])   # X axis points to Z
        y_axis = np.array([0, 1, 0])   # Y axis unchanged
        z_axis = np.array([-1, 0, 0])  # Z axis points to -X
        
        rotation_matrix = GXMLMath.get_plane_rotation_from_axis(x_axis, y_axis, z_axis)
        expected = np.array([[0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [-1, 0, 0, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "90° rotation around Y should match expected matrix")
        
        # Test with normalized arbitrary axes
        x_axis = GXMLMath.normalize(np.array([1, 1, 0]))   # 45 degrees in XY plane
        y_axis = GXMLMath.normalize(np.array([-1, 1, 0]))  # Perpendicular to xAxis in XY
        z_axis = np.array([0, 0, 1])                        # Z unchanged
        
        rotation_matrix = GXMLMath.get_plane_rotation_from_axis(x_axis, y_axis, z_axis)
        expected = np.array([[0.707107, 0.707107, 0, 0],
                           [-0.707107, 0.707107, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.assertTrue(np.allclose(rotation_matrix, expected),
                       "45° rotation in XY plane should match expected matrix")
        
        # Test that the resulting matrix properly transforms vectors
        original_vector = np.array([1, 0, 0])
        transformed_vector = GXMLMath.transform_direction(original_vector, rotation_matrix)
        self.assertTrue(np.allclose(transformed_vector, x_axis),
                       "Original X axis should transform to new X axis")


class GXMLMathHierarchyTests(unittest.TestCase):
    """Tests for hierarchical transform operations"""
    
    def testcombine_transforms(self):
        """Test combining child transform with parent transforms using different scale inheritance modes"""
        local_trans = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        parent_local_trans = GXMLMath.build_transform_matrix((3, 18, 42), (93, 64, 128), (94.17, 33.333, 14.5))
        parent_trans = GXMLMath.build_transform_matrix((93.87, 7.77, 66.66), (47.28, 87.92, 107.23), (16.2, 86.8, 154.8))
        parent_world_trans = GXMLMath.mat_mul(parent_local_trans, parent_trans)
        
        # Default scale inheritance
        actual = GXMLMath.combine_transform(local_trans, parent_world_trans, parent_local_trans, 
                                          GXMLMath.ScaleInheritance.Default)
        expected = np.array([[253.967, 10612.8, 1728.3, 0],
                           [29615.3, 36702.1, -482.52, 0],
                           [-40206.7, -16896.3, 1399.31, 0],
                           [-54068.3, -13852.1, 3324.26, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "Default scale inheritance should compound all transforms")
        
        # Offset only
        actual = GXMLMath.combine_transform(local_trans, parent_world_trans, parent_local_trans, 
                                          GXMLMath.ScaleInheritance.OffsetOnly)
        expected = np.array([[50.1451, 264.2, 68.907, 0],
                           [259.487, 454.663, -11.4562, 0],
                           [-771.444, 262.854, -17.0525, 0],
                           [-54068.3, -13852.1, 3324.26, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "OffsetOnly should inherit translation but not scale")
        
        # Offset and scale
        actual = GXMLMath.combine_transform(local_trans, parent_world_trans, parent_local_trans, 
                                          GXMLMath.ScaleInheritance.OffsetAndScale)
        expected = np.array([[4722.16, 24879.7, 6488.97, 0],
                           [8649.49, 15155.3, -381.87, 0],
                           [-11185.9, 3811.39, -247.261, 0],
                           [-54068.3, -13852.1, 3324.26, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "OffsetAndScale should inherit both translation and scale")
        
        # Scale only
        actual = GXMLMath.combine_transform(local_trans, parent_world_trans, parent_local_trans, 
                                          GXMLMath.ScaleInheritance.ScaleOnly)
        expected = np.array([[4722.16, 24879.7, 6488.97, 0],
                           [8649.49, 15155.3, -381.87, 0],
                           [-11185.9, 3811.39, -247.261, 0],
                           [1008.85, 6859.86, 249.773, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "ScaleOnly should inherit scale but compute translation from local offset")
        
        # Ignore
        actual = GXMLMath.combine_transform(local_trans, parent_world_trans, parent_local_trans, 
                                          GXMLMath.ScaleInheritance.Ignore)
        expected = np.array([[50.1451, 264.2, 68.907, 0],
                           [259.487, 454.663, -11.4562, 0],
                           [-771.444, 262.854, -17.0525, 0],
                           [1008.85, 6859.86, 249.773, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "Ignore should not inherit parent scale at all")
        
    def testcombine_transformsWithZeroScale(self):
        """Test combining transforms when parent has zero scale"""
        local_trans = GXMLMath.build_transform_matrix((2, 2, 2), (0, 0, 0), (1, 1, 1))
        parent_local_trans = parent_world_trans = GXMLMath.build_transform_matrix((1, 1, 1), (0, 0, 0), (0, 0, 0))
        
        actual = GXMLMath.combine_transform(local_trans, parent_world_trans, parent_local_trans, 
                                          GXMLMath.ScaleInheritance.Ignore)
        expected = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [3, 3, 3, 1]])
        self.assertTrue(np.allclose(actual, expected),
                       "Zero parent scale with Ignore mode should preserve child scale")


class GXMLMathVectorTests(unittest.TestCase):
    """Tests for vector operations"""
    
    def testDistance(self):
        """Test distance calculation between two points"""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        expected_distance = 5.0
        actual_distance = GXMLMath.distance(p1, p2)
        self.assertTrue(np.isclose(actual_distance, expected_distance),
                       "Distance from origin to (3,4,0) should be 5")
        
        # Test with 3D points
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 6.0, 8.0])
        expected_distance = np.sqrt(9 + 16 + 25)
        actual_distance = GXMLMath.distance(p1, p2)
        self.assertTrue(np.isclose(actual_distance, expected_distance),
                       "Distance should match 3D Euclidean distance")
    
    def testVectorLength(self):
        """Test vector length calculation"""
        vector = np.array([1.0, 2.0, 2.0])
        expected_length = 3.0
        actual_length = GXMLMath.length(vector)
        self.assertTrue(np.isclose(actual_length, expected_length),
                       "Vector [1, 2, 2] should have length 3")

    def testVectorNormalize(self):
        """Test vector normalization"""
        vector = np.array([1.0, 2.0, 2.0])
        expected_normalized = np.array([1.0/3.0, 2.0/3.0, 2.0/3.0])
        actual_normalized = GXMLMath.normalize(vector)
        self.assertTrue(np.allclose(actual_normalized, expected_normalized),
                       "Normalized vector should have unit length")
        
    def testVectorNormalizeZeroVector(self):
        """Test that normalizing zero vector raises ValueError"""
        vector = np.array([0.0, 0.0, 0.0])
        
        with self.assertRaises(ValueError):
            GXMLMath.normalize(vector)
    
    def testVectorsafe_normalize(self):
        """Test safe normalization that returns zero vector instead of raising error"""
        self.assertTrue(np.allclose(GXMLMath.safe_normalize(np.array((0, 0, 0))), (0, 0, 0)),
                       "Safe normalize of zero vector should return zero vector")

    def testVectorCrossProduct(self):
        """Test vector cross product"""
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = np.array([0.0, 1.0, 0.0])
        expected_cross_product = np.array([0.0, 0.0, 1.0])
        actual_cross_product = GXMLMath.cross(vector1, vector2)
        self.assertTrue(np.array_equal(actual_cross_product, expected_cross_product),
                       "X cross Y should equal Z")

    def testdot_product(self):
        """Test vector dot product"""
        vector1 = np.array([1.0, 2.0, 3.0])
        vector2 = np.array([4.0, 5.0, 6.0])
        expected_dot_product = 32.0
        actual_dot_product = GXMLMath.dot_product(vector1, vector2)
        self.assertTrue(np.isclose(actual_dot_product, expected_dot_product),
                       "Dot product should sum element-wise products")
    
    def testangle_between(self):
        """Test angle calculation between two vectors"""
        # Test perpendicular vectors (90 degrees)
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = np.array([0.0, 1.0, 0.0])
        angle = GXMLMath.angle_between(vector1, vector2)
        self.assertTrue(np.isclose(angle, 90.0),
                       "Perpendicular vectors should have 90° angle")
        
        # Test parallel vectors (0 degrees)
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = np.array([2.0, 0.0, 0.0])
        angle = GXMLMath.angle_between(vector1, vector2)
        self.assertTrue(np.isclose(angle, 0.0),
                       "Parallel vectors should have 0° angle")
        
        # Test opposite vectors (180 degrees)
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = np.array([-1.0, 0.0, 0.0])
        angle = GXMLMath.angle_between(vector1, vector2)
        self.assertTrue(np.isclose(angle, 180.0),
                       "Opposite vectors should have 180° angle")
        
        # Test 45-degree angle
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = GXMLMath.normalize(np.array([1.0, 1.0, 0.0]))
        angle = GXMLMath.angle_between(vector1, vector2)
        self.assertTrue(np.isclose(angle, 45.0),
                       "Vectors at 45° should return 45° angle")
        
    def testRotateVector(self):
        """Test rotating vector around axis"""
        # Rotate [1,0,0] around Y axis by 90 degrees should give [0,0,-1]
        vec = GXMLMath.rotate_vector([1, 0, 0], [0, 1, 0], 90)
        self.assertTrue(np.allclose(vec, [0, 0, -1]),
                       "Rotating X vector 90° around Y should point to -Z")
        
        # Rotate diagonal vector around Y axis by 90 degrees
        vec = GXMLMath.rotate_vector([0.70710678, 0.0, 0.70710678], [0, 1, 0], 90)
        self.assertTrue(np.allclose(vec, [0.70710678, 0.0, -0.70710678]),
                       "Rotating diagonal vector 90° around Y should rotate in XZ plane")
        
        # Rotate diagonal vector around Y axis by -90 degrees
        vec = GXMLMath.rotate_vector([0.70710678, 0.0, 0.70710678], [0, 1, 0], -90)
        self.assertTrue(np.allclose(vec, [-0.70710678, 0.0, 0.70710678]),
                       "Rotating diagonal vector -90° around Y should rotate opposite direction")


class GXMLMathTransformTests(unittest.TestCase):
    """Tests for point and direction transformations"""
    
    def testTransformPoint(self):
        """Test transforming point by matrix (applies translation)"""
        matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        point = (49.5, 87.4, 103.6)
        
        actual = GXMLMath.transform_point(point, matrix)
        expected = np.array([64.7856, 786.535, 719.96])
        
        self.assertTrue(np.allclose(actual, expected),
                       "Transformed point should apply full transform including translation")
    
    def testtransform_direction(self):
        """Test transforming direction vector by matrix (ignores translation)"""
        matrix = GXMLMath.build_transform_matrix((5, 9, 7), (23, 36, 92), (4.2, 3.7, 9.5))
        direction = (0.830057, 0.276686, 0.4842)
        
        actual = GXMLMath.transform_direction(direction, matrix)
        expected = np.array([-0.100255, 0.994163, -0.039847])
        
        self.assertTrue(np.allclose(actual, expected),
                       "Transformed direction should apply rotation/scale but not translation")


class GXMLMathIntersectionTests(unittest.TestCase):
    """Tests for geometric intersection calculations"""
    
    def testRayToSegmentIntersection(self):
        """Test ray-segment intersection detection"""
        intersection_point = GXMLMath.find_intersection_ray_to_segment((0, 0, 0), (1, 0, 0), 
                                                                   (1, 0, -1), (1, 0, 1))
        self.assertTrue(np.allclose(intersection_point, (1, 0, 0)),
                       "Ray along X should intersect segment at x=1")
        
    def testRayToSegmentIntersectionNoIntersection(self):
        """Test ray-segment when no intersection exists"""
        point = [0.5, 0, 0]
        direction = [0, 0, 1]
        p1 = [1, 0, -2]
        p2 = [0, 0, -2]
        intersection_point = GXMLMath.find_intersection_ray_to_segment(point, direction, p1, p2)
        self.assertIsNone(intersection_point,
                         "Ray parallel to segment should have no intersection")
        
    def testRayToRayIntersection(self):
        """Test ray-ray intersection detection"""
        # Test intersecting rays
        intersection_point = GXMLMath.find_intersection_between_rays((0, 0, 0), (1, 0, 0), 
                                                                  (2, 0, -2), (0, 0, -1))
        self.assertTrue(np.allclose(intersection_point, (2, 0, 0)),
                       "Two perpendicular rays should intersect at expected point")
        
        # Test parallel rays (no intersection)
        intersection_point = GXMLMath.find_intersection_between_rays((0, 0, 0), (1, 0, 0), 
                                                                  (0, 0, 2), (1, 0, 0))
        self.assertIsNone(intersection_point,
                         "Parallel rays should have no intersection")
        
        # Test collinear rays (no intersection)
        intersection_point = GXMLMath.find_intersection_between_rays((0, 0, 0), (1, 0, 0), 
                                                                  (5, 0, 0), (1, 0, 0))
        self.assertIsNone(intersection_point,
                         "Collinear rays should have no intersection")
    
    def testSegmentToSegmentIntersection(self):
        """Test segment-segment intersection detection"""
        # Test crossing segments in 3D
        p1 = [0, 0, 0]
        p2 = [2, 0, 0]
        p3 = [1, -1, 0]
        p4 = [1, 1, 0]
        intersection = GXMLMath.find_intersection_between_segments(p1, p2, p3, p4)
        self.assertTrue(np.allclose(intersection, [1, 0, 0]),
                       "Crossing segments should intersect at (1,0,0)")
        
        # Test non-intersecting segments (parallel)
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]
        p3 = [0, 1, 0]
        p4 = [1, 1, 0]
        intersection = GXMLMath.find_intersection_between_segments(p1, p2, p3, p4)
        self.assertIsNone(intersection,
                         "Parallel segments should have no intersection")
        
        # Test non-intersecting segments (skew in 3D)
        p1 = [0, 0, 0]
        p2 = [1, 0, 0]
        p3 = [0, 1, 1]
        p4 = [1, 1, 1]
        intersection = GXMLMath.find_intersection_between_segments(p1, p2, p3, p4)
        self.assertIsNone(intersection,
                         "Skew segments in 3D should have no intersection")
        
        # Test segments that don't reach intersection point
        p1 = [0, 0, 0]
        p2 = [0.5, 0, 0]
        p3 = [1, -1, 0]
        p4 = [1, 1, 0]
        intersection = GXMLMath.find_intersection_between_segments(p1, p2, p3, p4)
        self.assertIsNone(intersection,
                         "Segments that don't reach intersection should return None")
    
    def testintersect_line_with_plane(self):
        """Test ray-plane intersection"""
        # Test ray perpendicular to plane
        line_point = [0, 0, 0]
        line_direction = [0, 0, 1]
        plane_point = [0, 0, 5]
        plane_normal = [0, 0, 1]
        intersection = GXMLMath.intersect_line_with_plane(line_point, line_direction, 
                                                       plane_point, plane_normal)
        self.assertTrue(np.allclose(intersection, [0, 0, 5]),
                       "Ray perpendicular to plane should intersect at plane point")
        
        # Test ray at angle to plane
        line_point = [0, 0, 0]
        line_direction = [1, 0, 1]
        plane_point = [0, 0, 2]
        plane_normal = [0, 0, 1]
        intersection = GXMLMath.intersect_line_with_plane(line_point, line_direction, 
                                                       plane_point, plane_normal)
        self.assertTrue(np.allclose(intersection, [2, 0, 2]),
                       "Ray at 45° should intersect plane at expected point")
        
        # Test ray parallel to plane (no intersection)
        line_point = [0, 0, 0]
        line_direction = [1, 0, 0]
        plane_point = [0, 0, 1]
        plane_normal = [0, 0, 1]
        intersection = GXMLMath.intersect_line_with_plane(line_point, line_direction, 
                                                       plane_point, plane_normal)
        self.assertIsNone(intersection,
                         "Ray parallel to plane should have no intersection")
    
    def testproject_point_onto_plane(self):
        """Test projecting point onto plane"""
        # Test point above XY plane
        point = [2, 3, 5]
        plane_point = [0, 0, 0]
        plane_normal = [0, 0, 1]
        projected = GXMLMath.project_point_onto_plane(point, plane_point, plane_normal)
        self.assertTrue(np.allclose(projected, [2, 3, 0]),
                       "Point should project onto XY plane at z=0")
        
        # Test point already on plane
        point = [2, 3, 0]
        projected = GXMLMath.project_point_onto_plane(point, plane_point, plane_normal)
        self.assertTrue(np.allclose(projected, [2, 3, 0]),
                       "Point on plane should project to itself")
        
        # Test with tilted plane
        point = [0, 0, 0]
        plane_point = [1, 0, 0]
        plane_normal = [1, 0, 0]
        projected = GXMLMath.project_point_onto_plane(point, plane_point, plane_normal)
        self.assertTrue(np.allclose(projected, [1, 0, 0]),
                       "Origin should project onto x=1 plane")
    
    def testIsPointInsidePolygon(self):
        """Test point-in-polygon test on XZ plane"""
        # Test with square
        polygon = [
            [0, 0, 0],
            [2, 0, 0],
            [2, 0, 2],
            [0, 0, 2]
        ]
        
        # Point inside
        self.assertTrue(GXMLMath.is_point_inside_polygon([1, 0, 1], polygon),
                       "Point at center should be inside square")
        
        # Point outside
        self.assertFalse(GXMLMath.is_point_inside_polygon([3, 0, 1], polygon),
                        "Point outside should not be inside square")
        
        # Point on edge
        self.assertTrue(GXMLMath.is_point_inside_polygon([1, 0, 0], polygon),
                       "Point on edge should be considered inside")
        
        # Test with triangle
        polygon = [
            [0, 0, 0],
            [2, 0, 0],
            [1, 0, 2]
        ]
        
        self.assertTrue(GXMLMath.is_point_inside_polygon([1, 0, 0.5], polygon),
                       "Point inside triangle should be detected")
        self.assertFalse(GXMLMath.is_point_inside_polygon([0, 0, 2], polygon),
                        "Point outside triangle should not be inside")
    
    def testfind_interpolated_point(self):
        """Test finding interpolation parameter for point on line segment"""
        # Test point at start
        p1 = [0, 0, 0]
        p2 = [10, 0, 0]
        point = [0, 0, 0]
        t = GXMLMath.find_interpolated_point(point, p1, p2)
        self.assertTrue(np.isclose(t, 0.0),
                       "Point at start should have t=0")
        
        # Test point at end
        point = [10, 0, 0]
        t = GXMLMath.find_interpolated_point(point, p1, p2)
        self.assertTrue(np.isclose(t, 1.0),
                       "Point at end should have t=1")
        
        # Test point at midpoint
        point = [5, 0, 0]
        t = GXMLMath.find_interpolated_point(point, p1, p2)
        self.assertTrue(np.isclose(t, 0.5),
                       "Point at midpoint should have t=0.5")
        
        # Test point beyond segment
        point = [15, 0, 0]
        t = GXMLMath.find_interpolated_point(point, p1, p2)
        self.assertTrue(np.isclose(t, 1.5),
                       "Point beyond end should have t>1")
        
        # Test 3D segment
        p1 = [1, 2, 3]
        p2 = [4, 6, 8]
        point = [2.5, 4, 5.5]
        t = GXMLMath.find_interpolated_point(point, p1, p2)
        self.assertTrue(np.isclose(t, 0.5),
                       "Point at midpoint of 3D segment should have t=0.5")
    
    def testLerp(self):
        """Test linear interpolation"""
        # Test scalar lerp
        self.assertTrue(np.isclose(GXMLMath.lerp(0.0, 10, 20), 10),
                       "lerp at t=0 should return start value")
        self.assertTrue(np.isclose(GXMLMath.lerp(1.0, 10, 20), 20),
                       "lerp at t=1 should return end value")
        self.assertTrue(np.isclose(GXMLMath.lerp(0.5, 10, 20), 15),
                       "lerp at t=0.5 should return midpoint")
        self.assertTrue(np.isclose(GXMLMath.lerp(0.25, 0, 100), 25),
                       "lerp at t=0.25 should return 25% of range")
        
        # Test extrapolation
        self.assertTrue(np.isclose(GXMLMath.lerp(1.5, 10, 20), 25),
                       "lerp at t=1.5 should extrapolate beyond end")
        self.assertTrue(np.isclose(GXMLMath.lerp(-0.5, 10, 20), 5),
                       "lerp at t=-0.5 should extrapolate before start")
        
        # Test with numpy arrays
        result = GXMLMath.lerp(0.5, np.array([0, 10, 20]), np.array([10, 20, 40]))
        expected = np.array([5, 15, 30])
        self.assertTrue(np.allclose(result, expected),
                       "lerp should work with numpy arrays")


class GXMLMathUtilityTests(unittest.TestCase):
    """Tests for utility functions"""
    
    def testunpack_args(self):
        """Test unpacking tuple or individual arguments"""
        self.assertEqual(GXMLMath.unpack_args((1, 2, 3)), (1, 2, 3),
                        "Tuple argument should be unpacked")
        self.assertEqual(GXMLMath.unpack_args(1, 2, 3), (1, 2, 3),
                        "Individual arguments should be collected into tuple")
        
        with self.assertRaises(ValueError):
            GXMLMath.unpack_args(1, 2, 3, 4)
            
        def test_function_passing(*args, dummy="dummy"):
            """Helper to test passing unpacked args through function"""
            self.assertEqual(GXMLMath.unpack_args(*args), (1, 2, 3))
            
        test_function_passing((1, 2, 3))
        test_function_passing((1, 2, 3), dummy=42)
        
        with self.assertRaises(ValueError):
            # Second optional argument must be passed as named argument
            test_function_passing((1, 2, 3), 42)


class GXMLMathCreateTransformFromQuadTests(unittest.TestCase):
    """Tests for creating transform matrix from quad corner points"""
    
    def testIdentityCase(self):
        """Test with unit square at origin - should produce identity matrix"""
        world_points = [
            [0, 0, 0],  # p0: bottom-left
            [1, 0, 0],  # p1: bottom-right
            [1, 1, 0],  # p2: top-right
            [0, 1, 0]   # p3: top-left
        ]
        
        matrix = GXMLMath.create_transform_matrix_from_quad(world_points)
        expected = GXMLMath.identity()
        
        self.assertTrue(np.allclose(matrix, expected),
                       "Unit square at origin should produce identity matrix")
    
    def testTranslationOnly(self):
        """Test translation without rotation or scaling"""
        world_points = [
            [5, 3, 2],  # p0: bottom-left (translated origin)
            [6, 3, 2],  # p1: bottom-right
            [6, 4, 2],  # p2: top-right
            [5, 4, 2]   # p3: top-left
        ]
        
        matrix = GXMLMath.create_transform_matrix_from_quad(world_points)
        expected = GXMLMath.translate_matrix((5, 3, 2))
        
        self.assertTrue(np.allclose(matrix, expected),
                       "Translated unit square should produce translation matrix")
    
    def testScalingOnly(self):
        """Test scaling without rotation or translation"""
        world_points = [
            [0, 0, 0],  # p0: bottom-left
            [3, 0, 0],  # p1: bottom-right (3x scale in X)
            [3, 2, 0],  # p2: top-right
            [0, 2, 0]   # p3: top-left (2x scale in Y)
        ]
        
        matrix = GXMLMath.create_transform_matrix_from_quad(world_points)
        expected = GXMLMath.scale_matrix((3, 2, 1))
        
        self.assertTrue(np.allclose(matrix, expected),
                       "Scaled unit square should produce scale matrix")
    
    def testRotation90Degrees(self):
        """Test 90-degree rotation around Z axis"""
        world_points = [
            [0, 0, 0],    # p0: bottom-left (origin)
            [0, 1, 0],    # p1: bottom-right: X-axis points to +Y
            [-1, 1, 0],   # p2: top-right
            [-1, 0, 0]    # p3: top-left: Y-axis points to -X
        ]
        
        matrix = GXMLMath.create_transform_matrix_from_quad(world_points)
        
        # Test that all 4 provided points map correctly using row-vector multiplication
        corners = [
            [0, 0, 0, 1],  # should map to world_points[0]
            [1, 0, 0, 1],  # should map to world_points[1]
            [1, 1, 0, 1],  # should map to world_points[2]
            [0, 1, 0, 1]   # should map to world_points[3]
        ]
        
        for i, corner in enumerate(corners):
            transformed = np.array(corner) @ matrix
            self.assertTrue(np.allclose(transformed[:3], world_points[i], rtol=1e-10),
                           f"Corner {i} should map to world_points[{i}]")
    
    def testCombinedTransformations(self):
        """Test combination of translation, rotation, and scaling"""
        # Create a 2x2 square rotated 45 degrees and translated to (5,5,0)
        cos45 = np.cos(np.radians(45))
        sin45 = np.sin(np.radians(45))
        
        world_points = [
            [5, 5, 0],                                               # p0: bottom-left
            [5 + 2*cos45, 5 + 2*sin45, 0],                          # p1: bottom-right
            [5 + 2*cos45 - 2*sin45, 5 + 2*sin45 + 2*cos45, 0],     # p2: top-right
            [5 - 2*sin45, 5 + 2*cos45, 0]                           # p3: top-left
        ]
        
        matrix = GXMLMath.create_transform_matrix_from_quad(world_points)
        
        # Test origin is preserved
        origin = np.array([0, 0, 0, 1])
        transformed = origin @ matrix
        self.assertTrue(np.allclose(transformed[:3], [5, 5, 0]),
                       "Origin should map to translation offset")
        
        # Test that the X axis direction is correct (normalized)
        unit_x = np.array([1, 0, 0, 0])  # direction vector
        transformed_x = unit_x @ matrix
        expected_x_dir = np.array([cos45, sin45, 0])
        self.assertTrue(np.allclose(GXMLMath.normalize(transformed_x[:3]), expected_x_dir),
                       "X axis should point in rotated direction")
    
    def testNonRectangularQuad(self):
        """Test with a skewed quadrilateral"""
        world_points = [
            [0, 0, 0],    # p0: bottom-left
            [2, 0, 0],    # p1: bottom-right
            [2.5, 1, 0],  # p2: top-right
            [0.5, 1, 0]   # p3: top-left (skewed)
        ]
        
        matrix = GXMLMath.create_transform_matrix_from_quad(world_points)
        
        # Test that all 4 provided points map correctly
        corners = [
            [0, 0, 0, 1],  # should map to world_points[0]
            [1, 0, 0, 1],  # should map to world_points[1]
            [1, 1, 0, 1],  # should map to world_points[2]
            [0, 1, 0, 1]   # should map to world_points[3]
        ]
        
        for i, corner in enumerate(corners):
            transformed = np.array(corner) @ matrix
            self.assertTrue(np.allclose(transformed[:3], world_points[i], rtol=1e-10),
                           f"Corner {i} of skewed quad should map correctly")
    
    def test3DCase(self):
        """Test with points in 3D space (not coplanar with XY plane)"""
        world_points = [
            [0, 0, 0],  # p0: bottom-left
            [1, 0, 1],  # p1: bottom-right (raised in Z)
            [1, 1, 2],  # p2: top-right
            [0, 1, 1]   # p3: top-left (raised in Z)
        ]
        
        matrix = GXMLMath.create_transform_matrix_from_quad(world_points)
        
        # Test that all 4 provided points map correctly
        unit_corners = [
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [0, 1, 0, 1]
        ]
        
        for i, corner in enumerate(unit_corners):
            transformed = np.array(corner) @ matrix
            self.assertTrue(np.allclose(transformed[:3], world_points[i], rtol=1e-10),
                           f"3D corner {i} should map correctly")
    
    def testInvalidInput(self):
        """Test error handling for invalid inputs"""
        with self.assertRaises(ValueError):
            # Too few points
            GXMLMath.create_transform_matrix_from_quad([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
            
        with self.assertRaises(ValueError):
            # Too many points
            GXMLMath.create_transform_matrix_from_quad([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    
    def testMatrixProperties(self):
        """Test that the resulting matrix has expected properties"""
        world_points = [
            [1, 2, 3],  # p0
            [4, 2, 3],  # p1
            [4, 5, 3],  # p2
            [1, 5, 3]   # p3
        ]
        
        matrix_tuple = GXMLMath.create_transform_matrix_from_quad(world_points)
        
        # Matrix should be 4x4 tuple-of-tuples
        self.assertEqual(len(matrix_tuple), 4,
                        "Transform matrix should have 4 rows")
        self.assertTrue(all(len(row) == 4 for row in matrix_tuple),
                        "Each row should have 4 elements")
        
        # Convert to numpy for remaining tests
        matrix = np.array(matrix_tuple)
        
        # Bottom row should contain translation values [1, 2, 3, 1] for row-major format
        self.assertTrue(np.allclose(matrix[3, :], [1, 2, 3, 1]),
                       "Bottom row should contain translation and homogeneous 1")
        
        # Determinant should not be zero (matrix should be invertible)
        det = np.linalg.det(matrix)
        self.assertFalse(np.isclose(det, 0),
                        "Matrix should be invertible (non-zero determinant)")
        
        # Test that the matrix preserves all 4 defining points using row-vector multiplication
        unit_points = np.array([
            [0, 0, 0, 1],  # p0
            [1, 0, 0, 1],  # p1
            [1, 1, 0, 1],  # p2
            [0, 1, 0, 1]   # p3
        ])
        
        transformed_points = unit_points @ matrix
        
        for i in range(4):
            self.assertTrue(np.allclose(transformed_points[i, :3], world_points[i]),
                           f"Transformed point {i} should match world_points[{i}]")


class MathRotationTests(unittest.TestCase):
    """Tests for rotation utility functions."""
    
    def testRotateVectorAroundAxis(self):
        """Test Rodrigues rotation formula implementation."""
        # Rotate [1,0,0] around Z-axis by 90 degrees
        vector = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = 90.0
        
        rotated = GXMLMath.rotate_vector(vector, axis, angle)
        
        # Should rotate to approximately [0,1,0]
        np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-6)

    def testRotateVectorAroundAxisMultiplicationOrder(self):
        """Test that multiplication order in Rodrigues formula is correct."""
        # Rotate [1,1,0] around Z-axis by 45 degrees
        vector = np.array([1.0, 1.0, 0.0])
        axis = np.array([0.0, 0.0, 1.0])
        angle = 45.0
        
        rotated = GXMLMath.rotate_vector(vector, axis, angle)
        
        # Result should maintain length
        original_length = np.linalg.norm(vector)
        rotated_length = np.linalg.norm(rotated)
        self.assertAlmostEqual(original_length, rotated_length, places=5)

    def testRotateVectorCrossProductTerm(self):
        """Test that cross product term in rotation formula is correctly computed."""
        # Rotation where cross product matters
        vector = np.array([1.0, 0.0, 0.0])
        axis = np.array([0.0, 1.0, 0.0])
        angle = 90.0
        
        rotated = GXMLMath.rotate_vector(vector, axis, angle)
        
        # [1,0,0] rotated 90° around Y-axis -> [0,0,-1]
        np.testing.assert_allclose(rotated, [0, 0, -1], atol=1e-6)


class MathGeometryTests(unittest.TestCase):
    """Tests for geometry utility functions."""
    
    def testPointInPolygonWindingNumber(self):
        """Test winding number algorithm for point-in-polygon."""
        # Square polygon
        polygon = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])
        
        # Point inside
        self.assertTrue(GXMLMath.is_point_inside_polygon([0.5, 0, 0.5], polygon))
        
        # Point outside
        self.assertFalse(GXMLMath.is_point_inside_polygon([2, 0, 2], polygon))

    def testPointInPolygonIsLeftCalculation(self):
        """Test the is_left helper function used in winding number calculation."""
        # Triangle
        polygon = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 1]])
        
        # Point inside triangle
        inside = GXMLMath.is_point_inside_polygon([0.5, 0, 0.3], polygon)
        self.assertTrue(inside)
        
        # Point on edge (boundary case)
        on_edge = GXMLMath.is_point_inside_polygon([0.5, 0, 0], polygon)
        self.assertTrue(on_edge)

    def testPointInPolygonMultiplicationInIsLeft(self):
        """Test that is_left multiplication is correct for edge crossing detection."""
        # Trapezoid
        polygon = np.array([[0, 0, 0], [2, 0, 0], [1.5, 0, 1], [0.5, 0, 1]])
        
        # Test various points
        self.assertTrue(GXMLMath.is_point_inside_polygon([1, 0, 0.5], polygon))
        self.assertFalse(GXMLMath.is_point_inside_polygon([0, 0, -1], polygon))
        self.assertFalse(GXMLMath.is_point_inside_polygon([3, 0, 0.5], polygon))


class QuadBilinearTests(unittest.TestCase):
    """Tests for bilinear interpolation on quads."""
    
    def testBilinearInterpolateAddition(self):
        """Test that bilinear interpolation correctly adds the Z offset."""
        from mathutils.quad_interpolator import QuadInterpolator
        
        # Create a quad at Z=0
        p0 = np.array([0, 0, 0])
        p1 = np.array([1, 0, 0])
        p2 = np.array([1, 1, 0])
        p3 = np.array([0, 1, 0])
        quad = QuadInterpolator(p0, p1, p2, p3)
        
        # Interpolate at center with Z offset
        point = quad.bilinear_interpolate_point([0.5, 0.5, 5.0])
        
        # Should preserve the Z offset (point[2] + p[2])
        self.assertAlmostEqual(point[2], 5.0, places=5)

    def testBilinearInterpolateAdditionNonZeroQuad(self):
        """Test Z addition when quad itself has non-zero Z values."""
        from mathutils.quad_interpolator import QuadInterpolator
        
        # Quad with varying Z
        p0 = np.array([0, 0, 1])
        p1 = np.array([1, 0, 2])
        p2 = np.array([1, 1, 2])
        p3 = np.array([0, 1, 1])
        quad = QuadInterpolator(p0, p1, p2, p3)
        
        # Interpolate with additional Z offset
        point = quad.bilinear_interpolate_point([0.5, 0.5, 3.0])
        
        # Interpolated Z at (0.5,0.5) = 1.5, plus offset 3.0 = 4.5
        self.assertAlmostEqual(point[2], 4.5, places=5)


class GXMLTransformTests(unittest.TestCase):
    """Tests for GXMLTransform class."""
    
    def testSceneToLocalBasic(self):
        """Test sceneToLocal converts world coordinates to local [0-1] range."""
        from gxml_types import GXMLTransform
        
        transform = GXMLTransform()
        transform.transformationMatrix = GXMLMath.build_transform_matrix([0, 0, 0], [0, 0, 0], [2, 3, 4])
        
        # Point at origin
        local = transform.sceneToLocal([0, 0, 0])
        np.testing.assert_allclose(local, [0, 0, 0], atol=1e-6)
        
        # Point at max extent
        local = transform.sceneToLocal([2, 3, 4])
        np.testing.assert_allclose(local, [1, 1, 1], atol=1e-6)
        
        # Point at half
        local = transform.sceneToLocal([1, 1.5, 2])
        np.testing.assert_allclose(local, [0.5, 0.5, 0.5], atol=1e-6)

    def testSceneToLocalWithTranslation(self):
        """Test sceneToLocal with translation offset."""
        from gxml_types import GXMLTransform
        
        transform = GXMLTransform()
        transform.transformationMatrix = GXMLMath.build_transform_matrix([10, 20, 30], [0, 0, 0], [5, 5, 5])
        
        # Local origin is at world [10, 20, 30]
        local = transform.sceneToLocal([10, 20, 30])
        np.testing.assert_allclose(local, [0, 0, 0], atol=1e-6)
        
        # Local max is at world [15, 25, 35]
        local = transform.sceneToLocal([15, 25, 35])
        np.testing.assert_allclose(local, [1, 1, 1], atol=1e-6)

    def testSceneToLocalDivisionByRange(self):
        """Test that sceneToLocal correctly divides by the unit range (1,1,1)."""
        from gxml_types import GXMLTransform
        
        transform = GXMLTransform()
        transform.transformationMatrix = GXMLMath.build_transform_matrix([5, 5, 5], [0, 0, 0], [10, 10, 10])
        
        # After inverse transform, dividing by (1,1,1) should give normalized coords
        local = transform.sceneToLocal([7.5, 7.5, 7.5])  # Midpoint
        np.testing.assert_allclose(local, [0.25, 0.25, 0.25], atol=1e-5)

    def testTransformPoint(self):
        """Test transformPoint applies transformation correctly."""
        from gxml_types import GXMLTransform
        
        transform = GXMLTransform()
        transform.transformationMatrix = GXMLMath.build_transform_matrix([10, 0, 0], [0, 0, 0], [2, 2, 2])
        
        # Local [0,0,0] -> world [10,0,0]
        world = transform.transform_point([0, 0, 0])
        np.testing.assert_allclose(world, [10, 0, 0], atol=1e-6)
        
        # Local [1,0,0] -> world [12,0,0] (scaled by 2, offset by 10)
        world = transform.transform_point([1, 0, 0])
        np.testing.assert_allclose(world, [12, 0, 0], atol=1e-6)


class GXMLRayTests(unittest.TestCase):
    """Tests for GXMLRay class."""
    
    def test_from_points_creates_ray(self):
        """from_points should create a ray with correct origin, direction, length."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([3, 0, 0]))
        
        np.testing.assert_allclose(ray.origin, [0, 0, 0])
        np.testing.assert_allclose(ray.direction, [1, 0, 0])
        self.assertAlmostEqual(ray.length, 3.0)
    
    def test_from_points_returns_none_for_degenerate(self):
        """from_points should return None if points are too close."""
        ray = GXMLRay.from_points(np.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertIsNone(ray)
    
    def test_from_points_normalizes_direction(self):
        """Direction should always be normalized."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([3, 4, 0]))
        
        np.testing.assert_allclose(ray.direction, [0.6, 0.8, 0])
        self.assertAlmostEqual(np.linalg.norm(ray.direction), 1.0)
        self.assertAlmostEqual(ray.length, 5.0)
    
    def test_point_at_t_zero_returns_origin(self):
        """point_at_t(0) should return the origin."""
        ray = GXMLRay.from_points(np.array([1, 2, 3]), np.array([4, 5, 6]))
        
        np.testing.assert_allclose(ray.point_at_t(0), [1, 2, 3])
    
    def test_point_at_t_one_returns_end(self):
        """point_at_t(1) should return the end point."""
        ray = GXMLRay.from_points(np.array([1, 2, 3]), np.array([4, 5, 6]))
        
        np.testing.assert_allclose(ray.point_at_t(1), [4, 5, 6])
    
    def test_point_at_t_half_returns_midpoint(self):
        """point_at_t(0.5) should return the midpoint."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([10, 0, 0]))
        
        np.testing.assert_allclose(ray.point_at_t(0.5), [5, 0, 0])
    
    def test_point_at_t_beyond_one(self):
        """point_at_t can extrapolate beyond 1."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([10, 0, 0]))
        
        np.testing.assert_allclose(ray.point_at_t(2.0), [20, 0, 0])
    
    def test_point_at_t_negative(self):
        """point_at_t can extrapolate before 0."""
        ray = GXMLRay.from_points(np.array([10, 0, 0]), np.array([20, 0, 0]))
        
        np.testing.assert_allclose(ray.point_at_t(-0.5), [5, 0, 0])
    
    def test_project_point_on_ray(self):
        """project_point should return correct t for point on ray."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([10, 0, 0]))
        
        self.assertAlmostEqual(ray.project_point(np.array([0, 0, 0])), 0.0)
        self.assertAlmostEqual(ray.project_point(np.array([5, 0, 0])), 0.5)
        self.assertAlmostEqual(ray.project_point(np.array([10, 0, 0])), 1.0)
    
    def test_project_point_off_ray(self):
        """project_point should project perpendicular points onto ray."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([10, 0, 0]))
        
        # Point above the midpoint should project to t=0.5
        self.assertAlmostEqual(ray.project_point(np.array([5, 100, 0])), 0.5)
        self.assertAlmostEqual(ray.project_point(np.array([5, 0, 100])), 0.5)
    
    def test_project_point_beyond_ray(self):
        """project_point can return values outside 0-1 range."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([10, 0, 0]))
        
        self.assertAlmostEqual(ray.project_point(np.array([20, 0, 0])), 2.0)
        self.assertAlmostEqual(ray.project_point(np.array([-5, 0, 0])), -0.5)
    
    def test_roundtrip_point_at_t_and_project(self):
        """point_at_t and project_point should be inverses."""
        ray = GXMLRay.from_points(np.array([1, 2, 3]), np.array([4, 6, 8]))
        
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            point = ray.point_at_t(t)
            recovered_t = ray.project_point(point)
            self.assertAlmostEqual(recovered_t, t)
    
    def test_diagonal_ray(self):
        """Ray along diagonal should work correctly."""
        ray = GXMLRay.from_points(np.array([0, 0, 0]), np.array([1, 1, 1]))
        
        expected_length = np.sqrt(3)
        self.assertAlmostEqual(ray.length, expected_length)
        
        expected_dir = np.array([1, 1, 1]) / expected_length
        np.testing.assert_allclose(ray.direction, expected_dir)
        
        np.testing.assert_allclose(ray.point_at_t(0.5), [0.5, 0.5, 0.5])


class IntersectLines2DTests(unittest.TestCase):
    """Tests for intersect_lines_2d function."""
    
    def test_perpendicular_lines_xz_plane(self):
        """Two perpendicular lines in XZ plane should intersect."""
        # Line along X axis at z=0
        line1 = (np.array([0, 0, 0]), np.array([10, 0, 0]))
        # Line along Z axis at x=5
        line2 = (np.array([5, 0, -5]), np.array([5, 0, 5]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2, plane='xz')
        
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [5, 0, 0])
    
    def test_perpendicular_lines_xy_plane(self):
        """Two perpendicular lines in XY plane should intersect."""
        # Line along X axis at y=0
        line1 = (np.array([0, 0, 0]), np.array([10, 0, 0]))
        # Line along Y axis at x=3
        line2 = (np.array([3, -5, 0]), np.array([3, 5, 0]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2, plane='xy')
        
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [3, 0, 0])
    
    def test_perpendicular_lines_yz_plane(self):
        """Two perpendicular lines in YZ plane should intersect."""
        # Line along Y axis at z=0
        line1 = (np.array([0, 0, 0]), np.array([0, 10, 0]))
        # Line along Z axis at y=4
        line2 = (np.array([0, 4, -5]), np.array([0, 4, 5]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2, plane='yz')
        
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [0, 4, 0])
    
    def test_angled_lines(self):
        """Two angled lines should intersect at correct point."""
        # Diagonal line from origin
        line1 = (np.array([0, 0, 0]), np.array([10, 0, 10]))
        # Diagonal line crossing it
        line2 = (np.array([10, 0, 0]), np.array([0, 0, 10]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2, plane='xz')
        
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [5, 0, 5])
    
    def test_parallel_lines_return_none(self):
        """Parallel lines should return None."""
        line1 = (np.array([0, 0, 0]), np.array([10, 0, 0]))
        line2 = (np.array([0, 0, 5]), np.array([10, 0, 5]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2, plane='xz')
        
        self.assertIsNone(result)
    
    def test_y_value_from_first_line(self):
        """Result Y value should come from first point of line1."""
        line1 = (np.array([0, 7, 0]), np.array([10, 7, 0]))
        line2 = (np.array([5, 3, -5]), np.array([5, 3, 5]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2, plane='xz')
        
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[1], 7)  # Y from line1's first point
    
    def test_infinite_lines_intersect_beyond_segments(self):
        """Lines should intersect even if intersection is beyond segment endpoints."""
        # These segments don't overlap, but the infinite lines do
        line1 = (np.array([0, 0, 0]), np.array([2, 0, 0]))
        line2 = (np.array([10, 0, -1]), np.array([10, 0, 1]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2, plane='xz')
        
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [10, 0, 0])
    
    def test_default_plane_is_xz(self):
        """Default plane should be XZ."""
        line1 = (np.array([0, 5, 0]), np.array([10, 5, 0]))
        line2 = (np.array([5, 0, -5]), np.array([5, 0, 5]))
        
        result = GXMLMath.intersect_lines_2d(line1, line2)  # No plane specified
        
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [5, 5, 0])


if __name__ == '__main__':
    unittest.main()
