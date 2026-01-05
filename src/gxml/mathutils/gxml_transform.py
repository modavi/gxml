"""
GXMLTransform - Transformation matrix handling for GXML elements.
"""

from pathlib import Path
import mathutils.gxml_math as GXMLMath

# Try to use Mat4 C type for in-place operations
_HAS_MAT4 = False
Mat4 = None
try:
    from gxml.native_loader import load_native_extension
    _vec3 = load_native_extension('_vec3', Path(__file__).parent / 'native')
    if _vec3 is not None:
        Mat4 = _vec3.Mat4
        _HAS_MAT4 = True
except Exception:
    pass


class GXMLTransform:
    def __init__(self):
        self.pivot = (0.0, 0.0, 0.0)
        self.inheritScale = False
        
        # Use Mat4 C type if available for better performance
        if _HAS_MAT4:
            self.localTransformationMatrix = Mat4()  # Identity
            self.transformationMatrix = Mat4()  # Identity
            self._inverse_matrix = Mat4()  # Pre-allocated for caching
            self._inverse_valid = False
        else:
            self.localTransformationMatrix = GXMLMath.identity()
            self.transformationMatrix = GXMLMath.identity()
            self._inverse_matrix = None
            self._inverse_valid = False
        
        # We also store these components individually in the case of needing to know the original
        # values for translation and rotation when a 0 scale has been applied in any axis
        self.translation = (0.0, 0.0, 0.0)
        self.rotation = (0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)
    
    def __copy__(self):
        """Create a shallow copy with deep-copied Mat4 objects."""
        new_transform = GXMLTransform.__new__(GXMLTransform)
        new_transform.pivot = self.pivot
        new_transform.inheritScale = self.inheritScale
        new_transform.translation = self.translation
        new_transform.rotation = self.rotation
        new_transform.scale = self.scale
        
        if _HAS_MAT4:
            # Create new Mat4 objects and copy values
            new_transform.localTransformationMatrix = Mat4()
            new_transform.localTransformationMatrix.set_from(self.localTransformationMatrix)
            new_transform.transformationMatrix = Mat4()
            new_transform.transformationMatrix.set_from(self.transformationMatrix)
            new_transform._inverse_matrix = Mat4()
            new_transform._inverse_valid = False
        else:
            # Tuples are immutable, so shallow copy is fine
            new_transform.localTransformationMatrix = self.localTransformationMatrix
            new_transform.transformationMatrix = self.transformationMatrix
            new_transform._inverse_matrix = self._inverse_matrix
            new_transform._inverse_valid = self._inverse_valid
        
        return new_transform
    
    def set_local_matrix(self, matrix):
        """Set local transformation matrix from tuple or Mat4."""
        if _HAS_MAT4:
            if isinstance(matrix, Mat4):
                self.localTransformationMatrix.set_from(matrix)
            else:
                self.localTransformationMatrix.set_from(matrix)
        else:
            self.localTransformationMatrix = matrix
    
    def set_world_matrix(self, matrix):
        """Set world transformation matrix from tuple or Mat4."""
        if _HAS_MAT4:
            if isinstance(matrix, Mat4):
                self.transformationMatrix.set_from(matrix)
            else:
                self.transformationMatrix.set_from(matrix)
            self._inverse_valid = False
        else:
            self.transformationMatrix = matrix
            self._inverse_matrix = None
        
    def apply_local_transformations(self, translate, rotate, scale):
        if _HAS_MAT4:
            # Build TRS matrix directly in C
            self.localTransformationMatrix.set_trs(
                translate[0], translate[1], translate[2],
                rotate[0], rotate[1], rotate[2],
                scale[0], scale[1], scale[2]
            )
        else:
            self.localTransformationMatrix = GXMLMath.build_transform_matrix(translate, rotate, scale)
        self.translation = translate
        self.rotation = rotate
        self.scale = scale
        
    def decompose(self):
        return self.translation, self.rotation, self.scale
        
    def right(self):
        return self.transform_direction((1.0, 0.0, 0.0))

    def left(self):
        return self.transform_direction((-1.0, 0.0, 0.0))

    def up(self):
        return self.transform_direction((0.0, 1.0, 0.0))

    def down(self):
        return self.transform_direction((0.0, -1.0, 0.0))

    def forward(self):
        return self.transform_direction((0.0, 0.0, 1.0))

    def backward(self):
        return self.transform_direction((0.0, 0.0, -1.0))
    
    def sceneToLocal(self, point):
        if _HAS_MAT4:
            if not self._inverse_valid:
                # Copy current matrix to inverse and invert in-place
                self._inverse_matrix.set_from(self.transformationMatrix)
                try:
                    self._inverse_matrix.invert_into()
                except ValueError:
                    # Matrix is singular (zero scale), return origin
                    return [0, 0, 0]
                self._inverse_valid = True
            local_point = self._inverse_matrix.transform_point(point)
            return [local_point[0], local_point[1], local_point[2]]
        else:
            # Legacy path
            if self._inverse_matrix is None:
                det = GXMLMath.determinant(self.transformationMatrix)
                if abs(det) < 1e-10:
                    return [0,0,0]
                self._inverse_matrix = GXMLMath.invert(self.transformationMatrix)
            
            local_point = GXMLMath.transform_point(point, self._inverse_matrix)
            return [local_point[0], local_point[1], local_point[2]]

    def transform_point(self, point):
        if _HAS_MAT4:
            return self.transformationMatrix.transform_point(point)
        return GXMLMath.transform_point(point, self.transformationMatrix)
    
    def bilinear_transform_point(self, t, s, z_offset, quad_points):
        """Combined bilinear interpolation + matrix transform.
        
        Args:
            t: Interpolation parameter along width (0-1)
            s: Interpolation parameter along height (0-1)
            z_offset: Z offset to add after interpolation
            quad_points: Tuple of 4 corner points (p0, p1, p2, p3)
        
        Returns:
            Transformed point as Vec3 (if Mat4 available) or tuple
        """
        if _HAS_MAT4:
            return self.transformationMatrix.bilinear_transform(
                t, s, z_offset, 
                quad_points[0], quad_points[1], quad_points[2], quad_points[3]
            )
        # Fallback: do it in two steps using C extension bilinear_interpolate
        # _vec3 is already loaded at module level
        if _vec3 is not None:
            interp = _vec3.bilinear_interpolate(t, s, quad_points[0], quad_points[1],
                                          quad_points[2], quad_points[3])
            point = (interp[0], interp[1], z_offset + interp[2])
            return GXMLMath.transform_point(point, self.transformationMatrix)
        raise RuntimeError("_vec3 C extension required for bilinear_interpolate fallback")
    
    def transform_direction(self, vec):
        if _HAS_MAT4:
            return self.transformationMatrix.transform_direction(vec)
        return GXMLMath.transform_direction(vec, self.transformationMatrix)
    
    def recalculate(self, parentTransform):
        # Invalidate cached inverse since matrix is changing
        self._inverse_valid = False
        
        if _HAS_MAT4:
            # Build pivot-adjusted local matrix: translate(-pivot) @ localTransform
            # We'll use a temporary for the pivot translation
            if self.pivot != (0.0, 0.0, 0.0):
                # Need to pre-multiply by pivot translation
                pivot_mat = Mat4()
                pivot_mat.set_translation(-self.pivot[0], -self.pivot[1], -self.pivot[2])
                # localMatrix = pivot_mat @ localTransformationMatrix
                # Store result in transformationMatrix temporarily, then combine with parent
                self.transformationMatrix.multiply_into(pivot_mat, self.localTransformationMatrix)
                localMatrix = self.transformationMatrix
            else:
                localMatrix = self.localTransformationMatrix
                    
            if parentTransform:
                scaleInheritance = GXMLMath.ScaleInheritance.Default if self.inheritScale else GXMLMath.ScaleInheritance.Ignore
                
                parentWorldTransform = parentTransform.transformationMatrix
                parentLocalTransform = parentTransform.localTransformationMatrix
                
                # Check if any scale component is near zero
                ps = parentTransform.scale
                if abs(ps[0]) < 1e-10 or abs(ps[1]) < 1e-10 or abs(ps[2]) < 1e-10:
                    # Need to rebuild parent matrices without scale - fall back to old path for now
                    parentWorldTransform = GXMLMath.build_transform_matrix(parentTransform.translation, parentTransform.rotation, (1,1,1))
                    parentLocalTransform = GXMLMath.build_transform_matrix(parentTransform.translation, parentTransform.rotation, (1,1,1))
                    self.transformationMatrix.set_from(GXMLMath.combine_transform(localMatrix, parentWorldTransform, parentLocalTransform, scaleInheritance))
                elif scaleInheritance == GXMLMath.ScaleInheritance.Default:
                    # Fast path: just multiply local @ parent
                    self.transformationMatrix.multiply_into(localMatrix, parentWorldTransform)
                else:
                    # Complex scale inheritance - use existing Python path
                    self.transformationMatrix.set_from(GXMLMath.combine_transform(localMatrix, parentWorldTransform, parentLocalTransform, scaleInheritance))
            else:
                if localMatrix is not self.transformationMatrix:
                    self.transformationMatrix.set_from(localMatrix)
        else:
            # Legacy path
            if not hasattr(self, '_inverse_matrix') or self._inverse_matrix is None:
                pass  # Will be computed on demand
            self._inverse_matrix = None
            
            localMatrix = GXMLMath.mat_mul(GXMLMath.translate_matrix(tuple(-x for x in self.pivot)), self.localTransformationMatrix)
                    
            if(parentTransform):
                scaleInheritance = GXMLMath.ScaleInheritance.Default if self.inheritScale else GXMLMath.ScaleInheritance.Ignore
                
                parentWorldTransform = parentTransform.transformationMatrix
                parentLocalTransform = parentTransform.localTransformationMatrix
                
                # Check if any scale component is near zero (faster than np.isclose)
                ps = parentTransform.scale
                if abs(ps[0]) < 1e-10 or abs(ps[1]) < 1e-10 or abs(ps[2]) < 1e-10:
                    parentWorldTransform = GXMLMath.build_transform_matrix(parentTransform.translation, parentTransform.rotation, (1,1,1))
                    parentLocalTransform = GXMLMath.build_transform_matrix(parentTransform.translation, parentTransform.rotation, (1,1,1))
                
                self.transformationMatrix = GXMLMath.combine_transform(localMatrix, parentWorldTransform, parentLocalTransform, scaleInheritance)
            else:
                self.transformationMatrix = localMatrix
